"""Blok konfigurasi AUDIAX — satu-satunya sumber parameter numerik.

Kenapa ``frozen=True``: batasan rulebook #1 mewajibkan parameter model bersifat
*statis* saat demonstrasi berjalan. Dataclass yang tidak bisa dimutasi membuat
pelanggaran itu gagal keras (``FrozenInstanceError``) alih-alih diam-diam lolos.
Kalau butuh varian parameter (mis. ablasi denoise), pakai
``dataclasses.replace(cfg, ...)`` yang menghasilkan objek BARU — bukan mengubah
objek yang sedang dipakai proses inferensi.

Semua modul lain WAJIB membaca angka dari sini; jangan hardcode angka ajaib.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

__all__ = [
    "Config",
    "DEFAULT_CONFIG",
    "SCORING_BACKENDS",
    "SCHEMA_VERSION",
    "resolve_beats_source_dir",
    "resolve_beats_checkpoint",
    "resolve_adapter_checkpoint",
]

# Empat backend scoring resmi (CLAUDE.md "Keputusan Desain" #5). Urutan tuple
# ini tidak memengaruhi hasil (fusi memakai MIN), tapi dipertahankan stabil
# supaya `backend_stats` yang diserialisasi ke klien selalu punya kunci sama.
SCORING_BACKENDS: Tuple[str, ...] = ("cosine", "mahalanobis", "knn", "pca")

# Versi skema `MachineBaseline`. Naikkan kalau bentuk JSON-nya berubah supaya
# baseline lama di klien ditolak dengan pesan jelas, bukan salah dibaca.
SCHEMA_VERSION: str = "1.0"

# Root repo = induk dari paket `ai/`. Dipakai untuk mencari vendor source dan
# checkpoint tanpa memaksa pemanggil menyetel path absolut.
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_AI_ROOT: Path = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    """Parameter statis seluruh pipeline (Blok A–E).

    Nilai default di sini adalah nilai yang dipakai
    ``experiments/audiax_pipeline_AOT_New_MAC.ipynb`` Bagian 5 — kedua sisi wajib
    identik supaya tidak terjadi train/serve skew (CLAUDE.md §Aturan Wajib).
    """

    # ---------------- Blok A — quality gate ----------------
    sr: int = 16000
    min_duration_sec: float = 2.0
    max_clipping_ratio: float = 0.001
    min_snr_db: float = -10.0
    silence_rms: float = 1e-4

    # ---------------- Blok A — conditioning ----------------
    bp_low: float = 50.0
    bp_high: float = 7500.0
    bp_order: int = 4
    notch_low: float = 300.0        # pita suara manusia — diatenuasi demi privasi
    notch_high: float = 3400.0
    notch_order: int = 2

    # ---------------- Blok A — windowing ----------------
    window_sec: float = 2.0
    hop_sec: float = 1.0

    # ---------------- Blok A — denoise HPSS (opsional) ----------------
    # Default OFF, lihat CLAUDE.md "Keputusan Desain" #4: HPSS baru diuji pada
    # sinyal sintetis, belum pernah pada MIMII maupun rekaman blower asli.
    enable_denoise: bool = False
    denoise_harm_kernel: int = 31
    denoise_perc_kernel: int = 31
    denoise_mask_power: float = 2.0
    denoise_mask_floor: float = 0.1

    # ---------------- Blok B — encoder & adapter ----------------
    device: str = "cpu"             # produksi = CPU; GPU hanya untuk eksperimen
    encoder_batch_size: int = 8
    adapter_out_dim: int = 256

    # ---------------- Blok D — scoring ----------------
    backends: Tuple[str, ...] = SCORING_BACKENDS
    knn_k: int = 5
    pca_variance_target: float = 0.95
    mahalanobis_shrinkage: float = 0.1   # ala Ledoit-Wolf; N << D jadi wajib

    # ---------------- Blok C & E — kalibrasi & ambang ----------------
    z_warning: float = 3.0
    z_critical: float = 6.0
    min_calibration_windows: int = 20
    # Batas atas memory bank: menjaga ukuran payload baseline di klien dan
    # biaya O(N) scoring tetap terikat, walau operator merekam kelamaan.
    max_calibration_windows: int = 240
    # Jumlah window uji yang diagregasi saat inspeksi. Notebook Bagian 25
    # (`evaluate_machine`) — sumber seluruh angka AUC — merata-ratakan z
    # SELURUH window klip, bukan mengambil window pertama saja.
    max_inspect_windows: int = 16
    inspect_aggregation: str = "mean"    # "mean" | "max" | "first"

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.sr <= 0:
            raise ValueError("sr harus > 0")
        if self.window_sec <= 0 or self.hop_sec <= 0:
            raise ValueError("window_sec dan hop_sec harus > 0")
        if self.bp_low >= self.bp_high:
            raise ValueError("bp_low harus < bp_high")
        if self.notch_low >= self.notch_high:
            raise ValueError("notch_low harus < notch_high")
        if not 0.0 <= self.mahalanobis_shrinkage <= 1.0:
            raise ValueError("mahalanobis_shrinkage harus di [0, 1]")
        if not 0.0 < self.pca_variance_target <= 1.0:
            raise ValueError("pca_variance_target harus di (0, 1]")
        if self.knn_k < 1:
            raise ValueError("knn_k harus >= 1")
        if self.z_warning >= self.z_critical:
            raise ValueError("z_warning harus < z_critical")
        if self.min_calibration_windows < 2:
            raise ValueError("min_calibration_windows harus >= 2 (bank butuh >=2 titik)")
        if self.max_calibration_windows < self.min_calibration_windows:
            raise ValueError("max_calibration_windows harus >= min_calibration_windows")
        if self.max_inspect_windows < 1:
            raise ValueError("max_inspect_windows harus >= 1")
        if self.inspect_aggregation not in ("mean", "max", "first"):
            raise ValueError(
                "inspect_aggregation tidak dikenal: "
                + repr(self.inspect_aggregation)
                + " (pilihan: 'mean', 'max', 'first')"
            )
        unknown = [b for b in self.backends if b not in SCORING_BACKENDS]
        if unknown:
            raise ValueError(
                "backend tidak dikenal: " + repr(unknown) + "; sah: " + repr(list(SCORING_BACKENDS))
            )
        if len(self.backends) == 0:
            raise ValueError("minimal satu backend scoring harus aktif")

    # --- helper clamp: mencegah Wn di luar (0,1) saat fs rendah bikin butter() meledak ---

    def bp_clamped(self, fs: int) -> Tuple[float, float]:
        """Batas bandpass yang dijamin < Nyquist untuk ``fs`` apa pun."""
        nyq = 0.5 * fs
        return max(1.0, min(self.bp_low, nyq * 0.98)), min(self.bp_high, nyq * 0.99)

    def notch_clamped(self, fs: int) -> Tuple[float, float]:
        """Batas notch privasi yang dijamin < Nyquist untuk ``fs`` apa pun."""
        nyq = 0.5 * fs
        return max(1.0, min(self.notch_low, nyq * 0.95)), min(self.notch_high, nyq * 0.98)

    @property
    def window_samples(self) -> int:
        return int(self.window_sec * self.sr)

    @property
    def hop_samples(self) -> int:
        return int(self.hop_sec * self.sr)

    def fingerprint_fields(self) -> Tuple[Tuple[str, object], ...]:
        """Field yang mengubah ARTI sebuah embedding.

        Dipakai ``ai.calibration.model_fingerprint()``. Kalau salah satu berubah,
        memory bank lama tidak lagi sebanding dengan window uji baru, jadi
        baseline harus ditolak dan operator diminta kalibrasi ulang — bukan
        diam-diam dipakai dan menghasilkan z yang tidak berarti.
        """
        return (
            ("sr", self.sr),
            ("window_sec", self.window_sec),
            ("hop_sec", self.hop_sec),
            ("bp_low", self.bp_low),
            ("bp_high", self.bp_high),
            ("bp_order", self.bp_order),
            ("notch_low", self.notch_low),
            ("notch_high", self.notch_high),
            ("notch_order", self.notch_order),
            ("enable_denoise", self.enable_denoise),
            ("denoise_harm_kernel", self.denoise_harm_kernel),
            ("denoise_perc_kernel", self.denoise_perc_kernel),
            ("denoise_mask_power", self.denoise_mask_power),
            ("denoise_mask_floor", self.denoise_mask_floor),
        )


DEFAULT_CONFIG = Config()


# ---------------------------------------------------------------------------
# Resolusi lokasi artefak (vendor source & checkpoint)
#
# Kandidat dicoba berurutan supaya repo bisa dipakai apa adanya: layout resmi
# (`ai/vendor`, `ai/weights`) maupun layout kerja yang sudah ada di working tree
# (`third_party/unilm/beats`, `models/`). Override lewat environment variable
# selalu menang -- itu jalur yang dipakai Docker.
# ---------------------------------------------------------------------------

ENV_BEATS_SOURCE = "AUDIAX_BEATS_SOURCE"
ENV_BEATS_CHECKPOINT = "AUDIAX_BEATS_CHECKPOINT"
ENV_ADAPTER_CHECKPOINT = "AUDIAX_ADAPTER_CHECKPOINT"


def _first_existing(candidates: List[Path]) -> Optional[Path]:
    for c in candidates:
        if c.exists():
            return c
    return None


def _beats_source_candidates() -> List[Path]:
    env = os.environ.get(ENV_BEATS_SOURCE)
    out: List[Path] = [Path(env)] if env else []
    out += [
        _AI_ROOT / "vendor" / "beats",
        _REPO_ROOT / "third_party" / "unilm" / "beats",
    ]
    return out


def _beats_checkpoint_candidates() -> List[Path]:
    env = os.environ.get(ENV_BEATS_CHECKPOINT)
    out: List[Path] = [Path(env)] if env else []
    out += [
        _AI_ROOT / "weights" / "beats_finetuned.pt",
        _AI_ROOT / "weights" / "BEATs_iter3_plus_AS20K.pt",
        _REPO_ROOT / "models" / "BEATs_iter3_plus_AS20K.pt",
    ]
    return out


def resolve_beats_source_dir() -> Path:
    """Folder berisi ``BEATs.py``, ``backbone.py``, ``modules.py`` (di-vendor manual)."""
    for candidate in _beats_source_candidates():
        if (candidate / "BEATs.py").exists():
            return candidate
    raise FileNotFoundError(
        "Source BEATs tidak ditemukan (butuh BEATs.py). Vendor manual ke salah satu "
        "lokasi berikut, atau setel env " + ENV_BEATS_SOURCE + ": "
        + ", ".join(str(c) for c in _beats_source_candidates())
    )


def resolve_beats_checkpoint() -> Path:
    """Checkpoint BEATs yang dipakai encoder produksi.

    Urutan kandidat sengaja menaruh ``beats_finetuned.pt`` (hasil training MAC di
    notebook) paling depan, dan checkpoint pretrained mentah paling belakang.
    Fallback ke pretrained TIDAK disembunyikan: ``model_fingerprint`` ikut berubah
    mengikuti isi file, jadi baseline yang dibuat dengan satu checkpoint tidak
    akan pernah dipakai dengan checkpoint lain.
    """
    found = _first_existing(_beats_checkpoint_candidates())
    if found is None:
        raise FileNotFoundError(
            "Checkpoint BEATs tidak ditemukan. Taruh file .pt di salah satu lokasi "
            "berikut, atau setel env " + ENV_BEATS_CHECKPOINT + ": "
            + ", ".join(str(c) for c in _beats_checkpoint_candidates())
        )
    return found


def resolve_adapter_checkpoint() -> Optional[Path]:
    """Checkpoint adapter, atau ``None`` kalau belum ada.

    ``None`` bukan error: ``ai/__init__.py`` sengaja jatuh ke ``IdentityAdapter``
    supaya Blok C–E tetap bisa diuji end-to-end sebelum adapter selesai dilatih
    (lihat ``docs/model_implementation.md`` §8).
    """
    env = os.environ.get(ENV_ADAPTER_CHECKPOINT)
    candidates = [Path(env)] if env else []
    candidates.append(_AI_ROOT / "weights" / "adapter.pt")
    return _first_existing(candidates)
