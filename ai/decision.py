"""Blok E — keputusan, kartu kesehatan, dan atribusi fitur DSP.

Salinan logika ini WAJIB identik dengan
``experiments/audiax_pipeline_AOT_New_MAC.ipynb`` Bagian 22.

Batas klaim yang wajib dijaga (CLAUDE.md §"Yang sistem ini BUKAN"):

* Ini **alat bantu triase**, bukan diagnosis mengikat. ``HealthCard.disclaimer``
  selalu terisi dan tidak boleh disembunyikan di FE.
* Ini **bukan classifier jenis kerusakan**. ``dominant_indicator`` hanya
  menyebut fitur DSP mana yang paling bergeser dari kondisi kalibrasi — sebuah
  *petunjuk arah* untuk teknisi, bukan label kerusakan. MIMII sendiri tidak
  menyediakan taksonomi kerusakan berlabel, jadi klaim yang lebih kuat dari ini
  tidak bisa didukung data mana pun yang kita punya.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

from .calibration import MachineBaseline
from .config import Config, DEFAULT_CONFIG
from .scoring import fuse_scores, prepare_bank, score_against_prepared_bank

__all__ = [
    "HealthCard",
    "DISCLAIMER",
    "DSP_FEATURE_NAMES",
    "decide",
    "decide_windows",
    "dsp_features",
    "dsp_feature_stats",
    "compute_health_score",
]

DISCLAIMER = (
    "Alat bantu triase, bukan diagnosis mengikat -- tetap perlu inspeksi teknisi."
)

#: Urutan fitur DSP; indeksnya dipakai konsisten di seluruh modul.
DSP_FEATURE_NAMES: Tuple[str, ...] = ("kurtosis", "crest_factor", "spectral_centroid")

STATUS_NORMAL = "NORMAL"
STATUS_WARNING = "WARNING"
STATUS_CRITICAL = "CRITICAL"
STATUS_UNCALIBRATED = "KALIBRASI_KURANG"


@dataclass
class HealthCard:
    """Keluaran tunggal sistem, satu kartu per pemeriksaan.

    Attributes:
        status: ``NORMAL`` | ``WARNING`` | ``CRITICAL`` | ``KALIBRASI_KURANG``.
        z_score: Hasil fusi minimum, dalam satuan sigma mesin itu sendiri.
        health_score: 0–100, turunan monoton dari ``z_score`` untuk ditampilkan
            ke operator yang tidak paham satuan sigma. ``None`` kalau z tidak
            terdefinisi.
        calibration_quality: Diteruskan dari baseline.
        dominant_indicator: Fitur DSP yang paling bergeser, atau ``None``.
        disclaimer: Selalu terisi; wajib tampil di FE.
        reason: Penjelasan singkat kalau status ``KALIBRASI_KURANG``.
    """

    status: str
    z_score: float
    health_score: Optional[float]
    calibration_quality: str
    dominant_indicator: Optional[str]
    disclaimer: str
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Bentuk yang dikirim ``service/`` sebagai badan respons ``/v1/inspect``."""
        z = float(self.z_score)
        return {
            "status": self.status,
            "z_score": None if not np.isfinite(z) else round(z, 4),
            "health_score": None if self.health_score is None else round(float(self.health_score), 1),
            "calibration_quality": self.calibration_quality,
            "dominant_indicator": self.dominant_indicator,
            "disclaimer": self.disclaimer,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Atribusi fitur DSP
# ---------------------------------------------------------------------------


def dsp_features(w: np.ndarray, fs: int = DEFAULT_CONFIG.sr) -> np.ndarray:
    """Tiga fitur DSP klasik dari satu window waveform.

    * ``kurtosis`` — ketajaman puncak; naik saat muncul impuls (indikasi klasik
      cacat bantalan).
    * ``crest_factor`` — puncak dibagi RMS; naik pada benturan periodik.
    * ``spectral_centroid`` — pusat massa spektrum (Hz); bergeser saat energi
      pindah ke pita lain, mis. impeler kotor atau putaran tak seimbang.

    Ketiganya sengaja fitur yang bisa dijelaskan ke teknisi, bukan dimensi laten.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    kurt = float(scipy_stats.kurtosis(w))
    crest = float(np.max(np.abs(w)) / (np.sqrt(np.mean(w ** 2)) + 1e-12))
    spec = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(len(w), d=1.0 / float(fs))
    centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))
    return np.array([kurt, crest, centroid], dtype=np.float64)


def dsp_feature_stats(
    windows: Sequence[np.ndarray], fs: int = DEFAULT_CONFIG.sr
) -> Dict[str, List[float]]:
    """Mean/std tiap fitur DSP atas window kalibrasi.

    Disimpan ke ``MachineBaseline.notes`` supaya atribusi tetap bisa dihitung di
    jalur produksi yang **stateless**: server tidak menyimpan apa pun, dan klien
    hanya mengirim balik baseline JSON — bukan 2 menit audio kalibrasi. Yang
    disimpan cuma enam angka (statistik deskriptif), bukan data audio, jadi ini
    tetap di dalam batasan rulebook #3.
    """
    if len(windows) == 0:
        raise ValueError("Butuh minimal satu window kalibrasi untuk statistik DSP.")
    feats = np.array([dsp_features(w, fs) for w in windows], dtype=np.float64)
    return {
        "names": list(DSP_FEATURE_NAMES),
        "mu": [float(v) for v in feats.mean(axis=0)],
        "sigma": [float(v) for v in (feats.std(axis=0) + 1e-8)],
    }


def _attribute_from_stats(
    window: np.ndarray, fs: int, stats: Dict[str, Any]
) -> Optional[str]:
    """Fitur DSP dengan |z| terbesar terhadap statistik kalibrasi."""
    try:
        names = list(stats["names"])
        mu = np.asarray(stats["mu"], dtype=np.float64)
        sigma = np.asarray(stats["sigma"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if mu.shape != sigma.shape or mu.shape[0] != len(names):
        return None
    feat = dsp_features(window, fs)
    if feat.shape[0] != mu.shape[0]:
        return None
    z = np.abs((feat - mu) / np.maximum(sigma, 1e-8))
    if not np.all(np.isfinite(z)):
        return None
    return str(names[int(np.argmax(z))])


def _resolve_dsp_stats(
    baseline: MachineBaseline,
    calibration_windows: Optional[Sequence[np.ndarray]],
    fs: int,
) -> Optional[Dict[str, Any]]:
    """Statistik DSP dari window kalibrasi kalau ada, kalau tidak dari baseline."""
    if calibration_windows is not None and len(calibration_windows) > 0:
        return dsp_feature_stats(calibration_windows, fs)
    stats = baseline.notes.get("dsp_stats") if baseline.notes else None
    return stats if isinstance(stats, dict) else None


# ---------------------------------------------------------------------------
# Skor & keputusan
# ---------------------------------------------------------------------------


def compute_health_score(z: float, cfg: Config = DEFAULT_CONFIG) -> Optional[float]:
    """Petakan z ke 0–100 (100 = sehat), dipotong di kedua ujung.

    Linier terhadap z dan bukan probabilitas: kami tidak punya kalibrasi
    probabilistik yang bisa dipertanggungjawabkan, jadi menampilkan angka
    seolah-olah "peluang rusak 73%" akan menyesatkan. Ini murni skala tampilan.
    """
    if not np.isfinite(z):
        return None
    score = 100.0 * (1.0 - z / cfg.z_critical)
    return float(max(0.0, min(100.0, score)))


def _uncalibrated(quality: str, reason: str) -> HealthCard:
    return HealthCard(
        status=STATUS_UNCALIBRATED,
        z_score=float("nan"),
        health_score=None,
        calibration_quality=quality,
        dominant_indicator=None,
        disclaimer=DISCLAIMER,
        reason=reason,
    )


def decide_windows(
    windows: Sequence[np.ndarray],
    window_embeddings: np.ndarray,
    baseline: MachineBaseline,
    cfg: Config = DEFAULT_CONFIG,
    current_model_fingerprint: Optional[str] = None,
    calibration_windows: Optional[Sequence[np.ndarray]] = None,
) -> HealthCard:
    """Keputusan atas SATU klip uji yang sudah dipotong jadi beberapa window.

    Urutan pengecekan (``docs/model_implementation.md`` §9):

    1. ``model_fingerprint`` cocok? Kalau tidak → ``KALIBRASI_KURANG``.
    2. ``calibration_quality == "baik"``? Kalau tidak → ``KALIBRASI_KURANG``.
    3. Skor tiap window → fusi → agregasi antar-window → bandingkan ambang.
    4. Kalau bukan ``NORMAL``, hitung atribusi DSP pada window paling anomali.

    Agregasi antar-window mengikuti ``cfg.inspect_aggregation``. Default
    ``"mean"`` karena itulah yang dipakai ``evaluate_machine`` di notebook
    Bagian 25 — fungsi yang menghasilkan seluruh angka AUC yang dikutip. Memakai
    aturan lain di produksi berarti melaporkan angka yang tidak mengukur sistem
    yang benar-benar dijalankan.
    """
    if current_model_fingerprint is not None and baseline.model_fingerprint != current_model_fingerprint:
        return _uncalibrated(
            baseline.calibration_quality,
            "Baseline dibuat dengan model/parameter berbeda (" + baseline.model_fingerprint
            + " vs " + current_model_fingerprint + "). Lakukan kalibrasi ulang.",
        )

    if baseline.calibration_quality != "baik":
        return _uncalibrated(
            baseline.calibration_quality,
            "Rekaman kalibrasi terlalu pendek (" + str(baseline.n_windows) + " window, "
            "minimal " + str(cfg.min_calibration_windows) + "). Rekam ulang lebih lama.",
        )

    emb = np.asarray(window_embeddings, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb[None, :]
    if emb.shape[0] == 0:
        return _uncalibrated(
            baseline.calibration_quality,
            "Tidak ada window uji yang lolos quality gate.",
        )
    if len(windows) != emb.shape[0]:
        raise ValueError(
            "Jumlah window (" + str(len(windows)) + ") dan embedding (" + str(emb.shape[0])
            + ") tidak sama."
        )

    backend_names = tuple(baseline.backend_stats.keys())
    bank_model = prepare_bank(baseline.embeddings, cfg)

    z_per_window: List[float] = []
    for e in emb:
        raw = score_against_prepared_bank(e, bank_model, cfg, backend_names)
        z_per_window.append(fuse_scores(raw, baseline.backend_stats))

    z_arr = np.asarray(z_per_window, dtype=np.float64)
    if cfg.inspect_aggregation == "mean":
        z = float(z_arr.mean())
    elif cfg.inspect_aggregation == "max":
        z = float(z_arr.max())
    else:  # "first"
        z = float(z_arr[0])

    if z >= cfg.z_critical:
        status = STATUS_CRITICAL
    elif z >= cfg.z_warning:
        status = STATUS_WARNING
    else:
        status = STATUS_NORMAL

    dominant: Optional[str] = None
    if status != STATUS_NORMAL:
        stats = _resolve_dsp_stats(baseline, calibration_windows, cfg.sr)
        if stats is not None:
            worst = windows[int(np.argmax(z_arr))]
            dominant = _attribute_from_stats(np.asarray(worst), cfg.sr, stats)

    return HealthCard(
        status=status,
        z_score=z,
        health_score=compute_health_score(z, cfg),
        calibration_quality=baseline.calibration_quality,
        dominant_indicator=dominant,
        disclaimer=DISCLAIMER,
        reason=None,
    )


def decide(
    window: np.ndarray,
    window_embedding: np.ndarray,
    baseline: MachineBaseline,
    cfg: Config = DEFAULT_CONFIG,
    current_model_fingerprint: Optional[str] = None,
    calibration_windows: Optional[Sequence[np.ndarray]] = None,
) -> HealthCard:
    """Keputusan atas SATU window — kontrak seperti di ``docs/model_implementation.md`` §9.

    Sekadar kasus khusus :func:`decide_windows` dengan satu window, dipertahankan
    karena itulah bentuk yang dipakai walkthrough notebook Bagian 23.
    """
    return decide_windows(
        [np.asarray(window)],
        np.asarray(window_embedding).reshape(1, -1),
        baseline,
        cfg,
        current_model_fingerprint,
        calibration_windows,
    )
