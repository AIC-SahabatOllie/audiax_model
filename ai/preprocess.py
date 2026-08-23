"""Blok A — quality gate, conditioning, windowing, denoising (DSP klasik).

Salinan logika ini WAJIB identik dengan
``experiments/audiax_pipeline_AOT_New_MAC.ipynb`` Bagian 6. Notebook berdiri
sendiri (tidak ``import ai``), jadi kedua sisi bisa diam-diam menyimpang kalau
tidak disiplin — setiap perubahan di sini harus disinkronkan ke sana dan dicatat
di ``docs/PROGRESS.md`` (CLAUDE.md §Aturan Wajib).

Urutan wajib pipeline: resample → mono-mixdown → ``condition()`` → ``make_windows()``.
Urutan yang sama dipakai saat training maupun inferensi produksi; itulah yang
mencegah train/serve skew.

Modul ini murni NumPy/SciPy (+ torchaudio hanya untuk resample di ``load_audio``)
dan tidak boleh mengimpor framework web apa pun.
"""

from __future__ import annotations

import os
from typing import IO, List, Tuple, Union

import numpy as np
import scipy.ndimage as ndi
import scipy.signal as sig

from .config import Config, DEFAULT_CONFIG

__all__ = [
    "estimate_snr",
    "quality_gate",
    "bandpass",
    "privacy_notch",
    "peak_normalize",
    "hpss_denoise",
    "condition",
    "make_windows",
    "load_audio",
    "AudioSource",
]

#: Sumber audio yang diterima ``load_audio``: path di disk atau objek file
#: biner (dipakai ``service/`` untuk membaca ``UploadFile`` tanpa menulis
#: file sementara).
AudioSource = Union[str, os.PathLike, IO[bytes]]


def estimate_snr(x: np.ndarray, fs: int) -> float:
    """Estimasi SNR kasar dari rasio persentil RMS antar-frame 50 ms.

    Kenapa persentil dan bukan model derau sungguhan: tujuannya cuma menolak
    rekaman yang jelas rusak (kosong, gemuruh rata tanpa struktur) dengan biaya
    hampir nol. Frame keras (p90) dianggap sinyal, frame lirih (p10) dianggap
    lantai derau.
    """
    fl = max(int(0.05 * fs), 32)
    nf = max(len(x) // fl, 1)
    rms = np.sqrt(np.mean(x[: nf * fl].reshape(nf, fl) ** 2, axis=1) + 1e-12)
    return float(
        20 * np.log10((np.percentile(rms, 90) + 1e-12) / (np.percentile(rms, 10) + 1e-12))
    )


def quality_gate(
    x: np.ndarray, fs: int, cfg: Config = DEFAULT_CONFIG
) -> Tuple[bool, List[str], float]:
    """Blok A1 — tolak rekaman yang tidak layak dianalisis.

    Murah dan dijalankan lebih dulu supaya kegagalan demo yang paling sering
    (mikrofon mati, HP di saku, gain kelewat tinggi) muncul sebagai pesan jelas,
    bukan sebagai z-score acak.

    Returns:
        ``(lolos, daftar_alasan, snr_db)``. ``daftar_alasan`` kosong kalau lolos.
    """
    reasons: List[str] = []
    if len(x) / fs < cfg.min_duration_sec:
        reasons.append("short")
    rms = float(np.sqrt(np.mean(x ** 2) + 1e-12)) if len(x) else 0.0
    if rms < cfg.silence_rms:
        reasons.append("silent")
    cr = float(np.mean(np.abs(x) >= 0.999)) if len(x) else 1.0
    if cr > cfg.max_clipping_ratio:
        reasons.append("clipping")
    snr = estimate_snr(x, fs) if len(x) > 100 else -99.0
    if snr < cfg.min_snr_db:
        reasons.append("low_snr")
    return len(reasons) == 0, reasons, snr


def _filtfilt_safe(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``filtfilt`` yang tidak meledak pada sinyal sangat pendek.

    ``scipy.signal.filtfilt`` butuh ``len(x) > padlen``. Sinyal produksi selalu
    jauh lebih panjang, tapi unit test memakai potongan pendek — lebih baik
    melewatkan sinyal apa adanya daripada melempar ``ValueError`` yang tidak ada
    hubungannya dengan kualitas audio.
    """
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        return np.asarray(x, dtype=np.float64)
    return sig.filtfilt(b, a, x).astype(np.float64)


def bandpass(x: np.ndarray, fs: int, cfg: Config = DEFAULT_CONFIG) -> np.ndarray:
    """Blok A2 — Butterworth bandpass 50–7500 Hz (default).

    Batas bawah membuang rumble DC/handling, batas atas membuang pita yang di
    luar jangkauan mikrofon HP konsumer sekaligus menahan desis.
    """
    lo, hi = cfg.bp_clamped(fs)
    nyq = 0.5 * fs
    b, a = sig.butter(cfg.bp_order, [lo / nyq, hi / nyq], btype="band")
    return _filtfilt_safe(b, a, x)


def privacy_notch(x: np.ndarray, fs: int, cfg: Config = DEFAULT_CONFIG) -> np.ndarray:
    """Blok A2 — atenuasi pita suara manusia (300–3400 Hz).

    Perlindungan privasi operator: percakapan di sekitar mesin tidak ikut
    ter-embed. Konsekuensinya sebagian energi mesin di pita itu ikut hilang —
    trade-off ini disengaja dan berlaku sama di training maupun inferensi.
    """
    lo, hi = cfg.notch_clamped(fs)
    nyq = 0.5 * fs
    b, a = sig.butter(cfg.notch_order, [lo / nyq, hi / nyq], btype="bandstop")
    return _filtfilt_safe(b, a, x)


def peak_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Blok A2 — normalisasi puncak ke 0 dBFS, tahan DC offset/silence/NaN.

    Robustness-nya penting: jarak mikrofon dan gain otomatis HP berbeda tiap
    rekaman, jadi amplitudo absolut bukan informasi diagnostik dan harus
    dinormalkan sebelum embedding. ``nan_to_num`` + guard ``peak < eps`` mencegah
    pembagian nol pada rekaman senyap.
    """
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.mean(x)
    peak = np.max(np.abs(x)) if len(x) else 0.0
    if peak < eps:
        return x
    return (x / peak).astype(np.float64)


def hpss_denoise(x: np.ndarray, fs: int, cfg: Config = DEFAULT_CONFIG) -> np.ndarray:
    """Blok A2 (opsional) — Harmonic-Percussive Source Separation berbasis median.

    Menahan komponen harmonik (nada putaran blower) dan menekan komponen
    perkusif/broadband (derau lingkungan). **Default OFF** — lihat CLAUDE.md
    "Keputusan Desain" #4: efeknya baru terukur pada sinyal sintetis, belum pada
    MIMII maupun rekaman blower asli. Jangan jadikan default sebelum ada angka
    ablasi nyata dari notebook Bagian 29.
    """
    nperseg = 1024
    noverlap = nperseg - nperseg // 4
    if len(x) < nperseg:
        return np.asarray(x, dtype=np.float64)
    _, _, Zxx = sig.stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    harm = ndi.median_filter(mag, size=(1, cfg.denoise_harm_kernel), mode="reflect")
    perc = ndi.median_filter(mag, size=(cfg.denoise_perc_kernel, 1), mode="reflect")
    p = cfg.denoise_mask_power
    mask = (harm ** p) / (harm ** p + perc ** p + 1e-12)
    mask = cfg.denoise_mask_floor + (1.0 - cfg.denoise_mask_floor) * mask
    Zxx_denoised = mag * mask * np.exp(1j * phase)
    _, x_denoised = sig.istft(Zxx_denoised, fs=fs, nperseg=nperseg, noverlap=noverlap)
    if len(x_denoised) >= len(x):
        x_denoised = x_denoised[: len(x)]
    else:
        x_denoised = np.concatenate([x_denoised, np.zeros(len(x) - len(x_denoised))])
    return x_denoised.astype(np.float64)


def condition(x_raw: np.ndarray, fs: int, cfg: Config = DEFAULT_CONFIG) -> np.ndarray:
    """Orkestrasi Blok A2: bandpass → notch privasi → (opsional) denoise → normalize.

    Menerima array integer maupun float; integer dinormalkan ke [-1, 1] dulu
    supaya ambang ``quality_gate`` (yang berbasis skala penuh) tetap berarti.
    """
    x_raw = np.asarray(x_raw)
    x = x_raw.astype(np.float64)
    if np.issubdtype(x_raw.dtype, np.integer):
        x = x / np.iinfo(x_raw.dtype).max
    x = bandpass(x, fs, cfg)
    x = privacy_notch(x, fs, cfg)
    if cfg.enable_denoise:
        x = hpss_denoise(x, fs, cfg)
    return peak_normalize(x)


def make_windows(
    x: np.ndarray, fs: int, cfg: Config = DEFAULT_CONFIG
) -> List[np.ndarray]:
    """Blok A3 — potong jadi window ber-hop; tiap window lolos quality gate sendiri.

    Gate per-window (bukan per-klip) penting karena rekaman lapangan sering
    bagus di awal lalu tertutup tangan/kantong di tengah — window rusak dibuang,
    sisanya tetap terpakai.
    """
    win = int(cfg.window_sec * fs)
    hop = int(cfg.hop_sec * fs)
    if win <= 0 or hop <= 0 or len(x) < win:
        return []
    windows: List[np.ndarray] = []
    start = 0
    while start + win <= len(x):
        w = x[start : start + win]
        ok, _reasons, _snr = quality_gate(w, fs, cfg)
        if ok:
            windows.append(np.ascontiguousarray(w))
        start += hop
    return windows


def load_audio(source: AudioSource, cfg: Config = DEFAULT_CONFIG) -> np.ndarray:
    """Baca audio → mono → resample ke ``cfg.sr`` → float64 di [-1, 1].

    ``soundfile`` dicoba lebih dulu karena backend ``torchcodec`` di sebagian
    lingkungan (Colab, image slim) gagal memuat WAV biasa; ``torchaudio.load``
    dipakai sebagai cadangan. Keduanya menghasilkan array yang identik untuk WAV
    PCM, jadi pilihan backend tidak memengaruhi angka.
    """
    audio, srate = _read_any(source)

    # audio: [n_samples, n_channels] -> mono
    if audio.ndim == 2 and audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if audio.size == 0:
        raise ValueError("File audio kosong (0 sampel).")

    if srate != cfg.sr:
        audio = _resample(audio, srate, cfg.sr)

    return audio.astype(np.float64)


def _read_any(source: AudioSource) -> Tuple[np.ndarray, int]:
    """Baca file audio jadi ``(array [n, ch] float32, sample_rate)``."""
    try:
        import soundfile as sf  # opsional; lihat docstring load_audio

        data, srate = sf.read(source, dtype="float32", always_2d=True)
        return np.asarray(data), int(srate)
    except ImportError:
        pass
    except Exception as exc:  # file rusak / format tak didukung soundfile
        _sf_error = exc
        try:
            return _read_with_torchaudio(source)
        except Exception:
            raise ValueError("Gagal membaca file audio: " + str(_sf_error)) from _sf_error

    return _read_with_torchaudio(source)


def _read_with_torchaudio(source: AudioSource) -> Tuple[np.ndarray, int]:
    import torchaudio

    if hasattr(source, "seek"):
        source.seek(0)
    waveform, srate = torchaudio.load(source)  # [ch, n]
    return waveform.numpy().T, int(srate)


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample lewat torchaudio (kernel sinc berkualitas), fallback ke scipy."""
    try:
        import torch
        import torchaudio

        wav = torch.from_numpy(audio).unsqueeze(0)
        out = torchaudio.functional.resample(wav, src_sr, dst_sr)
        return out.squeeze(0).numpy()
    except Exception:
        n_out = int(round(len(audio) * dst_sr / float(src_sr)))
        return sig.resample(audio, n_out).astype(np.float32)
