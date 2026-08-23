"""Blok B — encoder audio (BEATs, beku).

BEATs adalah encoder pretrained AudioSet (Microsoft). Di AUDIAX ia dipakai
**beku total** saat inferensi: ``eval()`` + ``requires_grad_(False)`` +
``torch.inference_mode()``. Itu bukan sekadar optimasi kecepatan — batasan
rulebook #1 mewajibkan seluruh parameter statis selama demonstrasi berjalan,
jadi tidak boleh ada jalur kode apa pun di ``ai/`` yang bisa memutakhirkan bobot.

Checkpoint yang dipakai bebas (lihat ``docs/model_implementation.md`` §8):
pretrained mentah untuk tahap B0/B1, hasil fine-tuning MAC dari
``experiments/audiax_pipeline_AOT_New_MAC.ipynb`` Bagian 17 untuk B2. Yang
membedakan keduanya di sisi produksi hanyalah ``model_fingerprint`` — sebuah
baseline tidak pernah bisa dipakai lintas checkpoint.

Source BEATs **tidak** di-vendor lewat pip: file ``BEATs.py``/``backbone.py``/
``modules.py`` harus ada di folder yang dikembalikan
``ai.config.resolve_beats_source_dir()``.
"""

from __future__ import annotations

import hashlib
import sys
import threading
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .config import Config, DEFAULT_CONFIG, resolve_beats_checkpoint, resolve_beats_source_dir

__all__ = ["AudioEncoder", "BEATsEncoder", "file_sha256"]

_IMPORT_LOCK = threading.Lock()


@runtime_checkable
class AudioEncoder(Protocol):
    """Kontrak struktural encoder audio.

    Sengaja ``Protocol`` dan bukan ABC: ``DummyEncoder`` di ``tests/conftest.py``
    cukup punya method ``embed()`` dengan signature yang cocok, tanpa perlu
    mewarisi apa pun dan tanpa perlu torch sama sekali.
    """

    def embed(self, windows: np.ndarray) -> np.ndarray:
        """``[N, T]`` waveform float32 → ``[N, D]`` embedding ternormalisasi L2."""
        ...


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """SHA256 isi file. Dipakai sebagai identitas checkpoint di fingerprint."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _import_beats(source_dir: Path):
    """Impor modul BEATs dari folder vendor.

    ``BEATs.py`` memakai import absolut (``from backbone import ...``), jadi
    foldernya harus masuk ``sys.path``. Dikunci karena ``sys.path`` global dan
    ``warm_up()`` bisa dipanggil dari beberapa thread request sekaligus.
    """
    with _IMPORT_LOCK:
        resolved = str(source_dir.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        try:
            from BEATs import BEATs, BEATsConfig  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Gagal mengimpor source BEATs dari " + resolved + ". Pastikan BEATs.py, "
                "backbone.py, dan modules.py ada di folder itu."
            ) from exc
    return BEATs, BEATsConfig


class BEATsEncoder:
    """Encoder BEATs beku — satu forward pass per window.

    Embedding = rata-rata token encoder terakhir (mean pooling atas sumbu waktu),
    lalu dinormalisasi L2. Normalisasi L2 bukan kosmetik: backend ``cosine`` dan
    ``knn`` di Blok D mengasumsikan seluruh titik hidup di permukaan bola yang
    sama, sehingga jarak Euclidean dan jarak sudut jadi monoton satu sama lain.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        source_dir: Optional[str] = None,
        device: str = "cpu",
        cfg: Config = DEFAULT_CONFIG,
    ) -> None:
        import torch

        self.cfg = cfg
        self.device = device
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else resolve_beats_checkpoint()
        self.source_dir = Path(source_dir) if source_dir else resolve_beats_source_dir()

        BEATs, BEATsConfig = _import_beats(self.source_dir)

        checkpoint = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "cfg" not in checkpoint or "model" not in checkpoint:
            raise ValueError(
                "Format checkpoint BEATs tidak dikenal di " + str(self.checkpoint_path)
                + ": butuh dict dengan kunci 'cfg' dan 'model'."
            )

        beats_cfg = BEATsConfig(checkpoint["cfg"])
        model = BEATs(beats_cfg)
        model.load_state_dict(checkpoint["model"])

        # Checkpoint AudioSet yang di-fine-tune untuk klasifikasi tag punya
        # `predictor`; pada kasus itu `extract_features` mengembalikan probabilitas
        # 527 kelas, BUKAN fitur encoder 768-dim, dan mean-pooling di bawah akan
        # menghasilkan skalar tanpa peringatan. Tolak di depan.
        if getattr(model, "predictor", None) is not None:
            raise ValueError(
                "Checkpoint " + str(self.checkpoint_path) + " memuat predictor head "
                "(finetuned_model=True). extract_features() akan mengembalikan logit "
                "kelas AudioSet, bukan embedding encoder. Pakai checkpoint tanpa "
                "predictor (mis. hasil Bagian 17 notebook MAC)."
            )

        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model.to(device)
        self.beats_cfg = beats_cfg
        self.embed_dim: int = int(beats_cfg.encoder_embed_dim)

        # Identitas checkpoint: isi file, bukan nama file. Mengganti isi
        # `beats_finetuned.pt` tanpa mengganti namanya tetap membatalkan
        # baseline lama -- itu memang yang diinginkan.
        self.fingerprint: str = file_sha256(self.checkpoint_path)

    def embed(self, windows: np.ndarray) -> np.ndarray:
        """``[N, T]`` waveform → ``[N, D]`` embedding L2-normalized (float32).

        Dieksekusi di ``torch.inference_mode()``: tidak ada graf autograd yang
        dibangun, jadi tidak ada jalur untuk memutakhirkan bobot secara tak
        sengaja.
        """
        import torch

        arr = np.asarray(windows, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError("windows harus [N, T] atau [T], dapat shape " + str(arr.shape))
        if arr.shape[0] == 0:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        out = []
        bs = max(1, int(self.cfg.encoder_batch_size))
        with torch.inference_mode():
            for i in range(0, arr.shape[0], bs):
                batch = torch.from_numpy(arr[i : i + bs]).to(self.device)
                padding_mask = torch.zeros(batch.shape, dtype=torch.bool, device=batch.device)
                feats, _ = self.model.extract_features(batch, padding_mask=padding_mask)
                if feats.dim() != 3:
                    raise RuntimeError(
                        "extract_features mengembalikan tensor " + str(tuple(feats.shape))
                        + "; diharapkan [B, T, D]."
                    )
                emb = feats.mean(dim=1)
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)
                out.append(emb.float().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def __repr__(self) -> str:  # pragma: no cover - hanya untuk log
        return (
            "BEATsEncoder(checkpoint=" + self.checkpoint_path.name
            + ", dim=" + str(self.embed_dim)
            + ", device=" + self.device
            + ", fingerprint=" + self.fingerprint[:16] + ")"
        )
