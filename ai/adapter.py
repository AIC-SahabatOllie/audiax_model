"""Blok B lanjutan — adapter proyeksi (satu-satunya bobot yang pernah dilatih).

Adapter memproyeksikan embedding BEATs ``[N, 768]`` ke ruang yang lebih ringkas
``[N, 256]``. Motivasinya bukan kecepatan melainkan statistik: memory bank
kalibrasi berisi ~100–240 window, sementara dimensi BEATs 768. Backend
Mahalanobis dan PCA di Blok D memperkirakan struktur kovarians dari sampel
sebanyak itu — pada D=768 estimasinya sangat kurang tentu (N << D), pada D=256
jauh lebih stabil.

Dua kelas:

* ``Adapter``       — memuat ``ai/weights/adapter.pt``, bobot beku.
* ``IdentityAdapter`` — meneruskan embedding apa adanya (hanya renormalisasi L2).

``IdentityAdapter`` adalah FALLBACK yang disengaja: ``ai/__init__.py`` memakainya
otomatis kalau ``adapter.pt`` belum ada, supaya Blok C–E bisa diuji end-to-end
sebelum training selesai. Ia juga jalur yang benar untuk checkpoint hasil
``experiments/audiax_pipeline_AOT_New_MAC.ipynb``, karena notebook itu
mem-fine-tune backbone BEATs-nya langsung (dua layer encoder terakhir + head MAC
yang dibuang saat menyimpan) dan **tidak** menghasilkan ``adapter.pt``.

Sama seperti encoder: beku total, ``torch.inference_mode()``, tanpa gradien.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import Config, DEFAULT_CONFIG, resolve_adapter_checkpoint
from .encoder import file_sha256

__all__ = ["Adapter", "IdentityAdapter", "EmbeddingAdapter"]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Normalisasi L2 per-baris dengan guard vektor nol."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


class EmbeddingAdapter:
    """Kontrak minimal adapter: ``project([N, D_in]) -> [N, D_out]``.

    Dipakai sebagai type hint; ``IdentityAdapter``/``Adapter``/``DummyAdapter``
    di test semuanya memenuhi kontrak ini secara struktural.
    """

    fingerprint: str = "none"

    def project(self, embeddings: np.ndarray) -> np.ndarray:  # pragma: no cover - antarmuka
        raise NotImplementedError


class IdentityAdapter(EmbeddingAdapter):
    """Lewatkan embedding mentah, hanya renormalisasi L2.

    Renormalisasi tetap dilakukan supaya kontrak keluaran adapter selalu sama
    (``norm == 1``) apa pun implementasinya — Blok D bergantung pada itu.
    """

    fingerprint: str = "identity"

    def __init__(self, cfg: Config = DEFAULT_CONFIG) -> None:
        self.cfg = cfg

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[0] == 0:
            return arr.astype(np.float32)
        return _l2_normalize(arr)

    def __repr__(self) -> str:  # pragma: no cover - hanya untuk log
        return "IdentityAdapter()"


class Adapter(EmbeddingAdapter):
    """MLP proyeksi beku: ``Linear → BatchNorm1d → ReLU → Linear`` → L2-normalize.

    Dimensi TIDAK dihardcode: keduanya dibaca dari bentuk tensor di dalam
    checkpoint. Dengan begitu satu kelas ini bisa memuat adapter apa pun yang
    dilatih notebook tanpa perlu menyamakan ``Config`` secara manual — dan
    ketidakcocokan dimensi muncul saat memuat (pesan jelas), bukan saat scoring
    (angka diam-diam salah).

    Format checkpoint yang diharapkan::

        {"cfg": {...opsional...}, "model": state_dict}

    dengan kunci state_dict: ``fc1.weight``, ``fc1.bias``, ``bn.*``,
    ``fc2.weight``, ``fc2.bias``.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        cfg: Config = DEFAULT_CONFIG,
        device: str = "cpu",
    ) -> None:
        import torch
        import torch.nn as nn

        self.cfg = cfg
        self.device = device

        path = Path(checkpoint_path) if checkpoint_path else resolve_adapter_checkpoint()
        if path is None or not Path(path).exists():
            raise FileNotFoundError(
                "Checkpoint adapter tidak ditemukan. Pakai IdentityAdapter kalau adapter "
                "memang belum dilatih (lihat docs/model_implementation.md §8)."
            )
        self.checkpoint_path = Path(path)

        ckpt = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        if not isinstance(state, dict) or "fc1.weight" not in state or "fc2.weight" not in state:
            raise ValueError(
                "Format checkpoint adapter tidak dikenal di " + str(self.checkpoint_path)
                + ": butuh state_dict dengan kunci fc1.weight dan fc2.weight."
            )

        hidden_dim, in_dim = tuple(state["fc1.weight"].shape)
        out_dim, hidden_check = tuple(state["fc2.weight"].shape)
        if hidden_check != hidden_dim:
            raise ValueError(
                "Dimensi adapter tidak konsisten: fc1 keluar " + str(hidden_dim)
                + " tapi fc2 masuk " + str(hidden_check) + "."
            )

        module = nn.Sequential()
        module.add_module("fc1", nn.Linear(in_dim, hidden_dim))
        module.add_module("bn", nn.BatchNorm1d(hidden_dim))
        module.add_module("relu", nn.ReLU(inplace=True))
        module.add_module("fc2", nn.Linear(hidden_dim, out_dim))
        module.load_state_dict(state)
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

        self.module = module.to(device)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.fingerprint = file_sha256(self.checkpoint_path)

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        """``[N, in_dim]`` → ``[N, out_dim]`` float32 ternormalisasi L2."""
        import torch

        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[0] == 0:
            return np.zeros((0, self.out_dim), dtype=np.float32)
        if arr.shape[1] != self.in_dim:
            raise ValueError(
                "Dimensi embedding " + str(arr.shape[1]) + " tidak cocok dengan adapter "
                "(butuh " + str(self.in_dim) + "). Encoder dan adapter berasal dari "
                "eksperimen berbeda?"
            )

        outs: List[np.ndarray] = []
        bs = max(1, int(self.cfg.encoder_batch_size))
        with torch.inference_mode():
            for i in range(0, arr.shape[0], bs):
                chunk = torch.from_numpy(arr[i : i + bs]).to(self.device)
                # BatchNorm dalam mode eval() memakai running stats, jadi hasil
                # per-sampel tidak bergantung pada isi batch -- penting supaya
                # skor kalibrasi dan skor inspeksi sebanding.
                outs.append(self.module(chunk).float().cpu().numpy())
        return _l2_normalize(np.concatenate(outs, axis=0))

    def __repr__(self) -> str:  # pragma: no cover - hanya untuk log
        return (
            "Adapter(" + str(self.in_dim) + "->" + str(self.out_dim)
            + ", fingerprint=" + self.fingerprint[:16] + ")"
        )
