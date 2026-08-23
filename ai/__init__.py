"""AUDIAX — paket inti AI (pure Python, NOL dependensi web).

API publik hanya tiga fungsi::

    from ai import calibrate, inspect, warm_up

    baseline = calibrate("kalibrasi.wav", machine_label="Blower Oven 1")  # ~2 menit
    card = inspect("uji.wav", baseline=baseline)                          # ~10 detik
    warm_up()   # muat encoder+adapter ke memori sekarang (dipanggil /healthz)

Kontrak arsitektural yang ditegakkan modul ini:

* **Model dimuat sebagai singleton**, sekali per proses, dipakai ulang untuk
  semua permintaan berikutnya. Memuat BEATs 360 MB per-request akan membuat
  latensi tidak masuk akal dan memori meledak.
* **Nol state selain model.** ``MachineBaseline`` dikembalikan ke pemanggil dan
  disimpan di klien; tidak ada cache, tidak ada database, tidak ada file yang
  ditulis (batasan rulebook #3).
* **Parameter statis.** Seluruh bobot beku; kalibrasi hanya menghitung statistik
  deskriptif (batasan rulebook #1 & #2).

Parameter berawalan garis bawah (``_encoder_override``, ``_adapter_override``)
HANYA untuk testing dengan komponen tiruan — tidak pernah dipakai di jalur
produksi.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .adapter import Adapter, EmbeddingAdapter, IdentityAdapter
from .calibration import MachineBaseline, build_baseline, model_fingerprint
from .config import (
    Config,
    DEFAULT_CONFIG,
    SCHEMA_VERSION,
    SCORING_BACKENDS,
    resolve_adapter_checkpoint,
)
from .decision import DISCLAIMER, HealthCard, decide, decide_windows, dsp_feature_stats
from .encoder import AudioEncoder, BEATsEncoder
from .preprocess import AudioSource, condition, load_audio, make_windows, quality_gate
from .scoring import compute_all_backend_scores, fuse_scores

__all__ = [
    # --- API publik utama (docs/model_implementation.md §4) ---
    "calibrate",
    "inspect",
    "warm_up",
    "model_info",
    # --- tipe data ---
    "Config",
    "DEFAULT_CONFIG",
    "MachineBaseline",
    "HealthCard",
    "DISCLAIMER",
    "SCHEMA_VERSION",
    "SCORING_BACKENDS",
    "AudioSource",
    # --- komponen model ---
    "AudioEncoder",
    "BEATsEncoder",
    "Adapter",
    "IdentityAdapter",
    # --- Blok A-E, diekspor ulang supaya service/ dan pemeriksaan paritas
    #     notebook tidak perlu mengimpor submodul satu per satu ---
    "quality_gate",
    "condition",
    "make_windows",
    "load_audio",
    "build_baseline",
    "model_fingerprint",
    "compute_all_backend_scores",
    "fuse_scores",
    "decide",
    "decide_windows",
    "dsp_feature_stats",
]

__version__ = "1.0.0"

# --- singleton model (satu per proses) -------------------------------------
_LOAD_LOCK = threading.Lock()
_ENCODER: Optional[AudioEncoder] = None
_ADAPTER: Optional[EmbeddingAdapter] = None
_ADAPTER_KIND: str = "belum dimuat"


def _ensure_models_loaded(cfg: Config) -> Tuple[AudioEncoder, EmbeddingAdapter]:
    """Muat encoder+adapter sekali, lalu pakai ulang.

    Jatuh otomatis ke :class:`IdentityAdapter` kalau ``ai/weights/adapter.pt``
    belum ada. Itu bukan kegagalan diam-diam: ``model_info()`` melaporkan
    adapter mana yang aktif, dan ``model_fingerprint`` berbeda antara keduanya,
    jadi baseline yang dibuat dengan Identity tidak akan pernah dipakai dengan
    Adapter terlatih (dan sebaliknya).
    """
    global _ENCODER, _ADAPTER, _ADAPTER_KIND

    if _ENCODER is not None and _ADAPTER is not None:
        return _ENCODER, _ADAPTER

    with _LOAD_LOCK:
        if _ENCODER is None:
            _ENCODER = BEATsEncoder(device=cfg.device, cfg=cfg)
        if _ADAPTER is None:
            adapter_path = resolve_adapter_checkpoint()
            if adapter_path is None:
                _ADAPTER = IdentityAdapter(cfg)
                _ADAPTER_KIND = "IdentityAdapter (adapter.pt belum ada)"
            else:
                _ADAPTER = Adapter(str(adapter_path), cfg=cfg, device=cfg.device)
                _ADAPTER_KIND = "Adapter (" + adapter_path.name + ")"
    return _ENCODER, _ADAPTER


def _resolve_models(
    cfg: Config,
    encoder_override: Optional[AudioEncoder],
    adapter_override: Optional[EmbeddingAdapter],
) -> Tuple[AudioEncoder, EmbeddingAdapter]:
    """Pilih model tiruan (test) atau singleton produksi.

    Kalau salah satu override diberikan, model produksi TIDAK dimuat sama
    sekali — itulah yang membuat ``tests/test_pipeline.py`` bisa berjalan tanpa
    checkpoint 360 MB.
    """
    if encoder_override is not None or adapter_override is not None:
        encoder = encoder_override if encoder_override is not None else _ENCODER
        adapter = adapter_override if adapter_override is not None else IdentityAdapter(cfg)
        if encoder is None:
            raise ValueError("_encoder_override wajib diisi kalau model asli tidak dimuat.")
        return encoder, adapter
    return _ensure_models_loaded(cfg)


def _load_and_window(
    source: AudioSource, cfg: Config, stage: str
) -> List[np.ndarray]:
    """resample+mono → ``condition()`` → ``make_windows()``.

    Urutan ini identik dengan notebook Bagian 6 dan dipakai di training maupun
    inferensi — itulah pertahanan terhadap train/serve skew.
    """
    raw = load_audio(source, cfg)

    ok, reasons, snr = quality_gate(raw, cfg.sr, cfg)
    if not ok:
        raise ValueError(
            "Audio " + stage + " tidak lolos quality gate (" + ", ".join(reasons) + "; "
            "SNR~" + format(snr, ".1f") + " dB). Rekam ulang lebih dekat ke mesin, "
            "pastikan mikrofon tidak tertutup, dan hindari volume berlebihan."
        )

    windows = make_windows(condition(raw, cfg.sr, cfg), cfg.sr, cfg)
    if not windows:
        raise ValueError(
            "Tidak ada window yang lolos quality gate dari audio " + stage + ". "
            "Durasi minimal " + format(cfg.window_sec, ".0f") + " detik dan sinyal harus "
            "cukup kuat."
        )
    return windows


def _embed(
    windows: Sequence[np.ndarray],
    encoder: AudioEncoder,
    adapter: EmbeddingAdapter,
) -> np.ndarray:
    """``[N, T]`` waveform → encoder → adapter → ``[N, D]`` embedding."""
    batch = np.stack([np.asarray(w, dtype=np.float32) for w in windows], axis=0)
    return np.asarray(adapter.project(encoder.embed(batch)), dtype=np.float32)


# ---------------------------------------------------------------------------
# API publik
# ---------------------------------------------------------------------------


def warm_up(cfg: Config = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Muat encoder+adapter ke memori sekarang.

    Dipanggil ``service/`` di ``/healthz`` supaya orkestrator (``depends_on:
    service_healthy``) tidak mengirim permintaan sebelum model benar-benar siap.
    Melempar kalau vendor source atau checkpoint tidak ada — gagal cepat dan
    jelas, bukan gagal di request pertama pengguna.
    """
    _ensure_models_loaded(cfg)
    return model_info(cfg)


def model_info(cfg: Config = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Ringkasan model yang sedang aktif — untuk log, ``/healthz``, dan debugging."""
    encoder, adapter = _ensure_models_loaded(cfg)
    return {
        "version": __version__,
        "schema_version": SCHEMA_VERSION,
        "encoder": type(encoder).__name__,
        "encoder_checkpoint": str(getattr(encoder, "checkpoint_path", "n/a")),
        "adapter": _ADAPTER_KIND,
        "embedding_dim": int(getattr(encoder, "embed_dim", 0)),
        "device": cfg.device,
        "backends": list(cfg.backends),
        "model_fingerprint": model_fingerprint(encoder, adapter, cfg),
        "enable_denoise": cfg.enable_denoise,
    }


def calibrate(
    audio_path: AudioSource,
    machine_label: str,
    cfg: Config = DEFAULT_CONFIG,
    _encoder_override: Optional[AudioEncoder] = None,
    _adapter_override: Optional[EmbeddingAdapter] = None,
) -> MachineBaseline:
    """Blok A→C — bangun baseline satu unit mesin dari rekaman kondisi sehat.

    Args:
        audio_path: Path atau objek file WAV rekaman kalibrasi (~2 menit).
        machine_label: Label bebas dari operator, mis. ``"Blower Oven 1"``.
        cfg: Parameter statis.

    Returns:
        :class:`MachineBaseline` — **dikembalikan ke pemanggil**, tidak disimpan
        di sisi ini. Klien yang menyimpannya dan mengirimkannya kembali setiap
        kali inspeksi.

    Raises:
        ValueError: Audio tidak lolos quality gate atau tidak menghasilkan
            window yang layak.
    """
    if not machine_label or not str(machine_label).strip():
        raise ValueError("machine_label wajib diisi.")

    encoder, adapter = _resolve_models(cfg, _encoder_override, _adapter_override)
    windows = _load_and_window(audio_path, cfg, "kalibrasi")
    embeddings = _embed(windows, encoder, adapter)

    baseline = build_baseline(
        embeddings=embeddings,
        machine_label=str(machine_label).strip(),
        model_fingerprint=model_fingerprint(encoder, adapter, cfg),
        cfg=cfg,
    )

    # Statistik DSP ikut dibawa supaya `dominant_indicator` tetap bisa dihitung
    # di jalur stateless -- server tidak pernah menyimpan audio kalibrasi.
    baseline.notes["dsp_stats"] = dsp_feature_stats(windows, cfg.sr)
    return baseline


def inspect(
    audio_path: AudioSource,
    baseline: MachineBaseline,
    cfg: Config = DEFAULT_CONFIG,
    _encoder_override: Optional[AudioEncoder] = None,
    _adapter_override: Optional[EmbeddingAdapter] = None,
) -> HealthCard:
    """Blok A→E — periksa satu rekaman harian terhadap baseline mesinnya.

    Args:
        audio_path: Path atau objek file WAV rekaman uji (~10 detik).
        baseline: Baseline unit itu, dikirim balik oleh klien.
        cfg: Parameter statis.

    Returns:
        :class:`HealthCard` berisi status, z-score, dan disclaimer triase.

    Raises:
        ValueError: Audio tidak lolos quality gate, atau baseline tidak sah.
    """
    if not isinstance(baseline, MachineBaseline):
        raise ValueError("baseline harus instance MachineBaseline.")

    encoder, adapter = _resolve_models(cfg, _encoder_override, _adapter_override)
    windows = _load_and_window(audio_path, cfg, "uji")

    # Batasi jumlah window yang diskor: rekaman 10 detik menghasilkan ~9 window,
    # tapi klien bisa saja mengirim rekaman jauh lebih panjang.
    if len(windows) > cfg.max_inspect_windows:
        idx = np.linspace(0, len(windows) - 1, cfg.max_inspect_windows).round().astype(int)
        windows = [windows[i] for i in sorted(set(idx.tolist()))]

    embeddings = _embed(windows, encoder, adapter)

    return decide_windows(
        windows=windows,
        window_embeddings=embeddings,
        baseline=baseline,
        cfg=cfg,
        current_model_fingerprint=model_fingerprint(encoder, adapter, cfg),
    )
