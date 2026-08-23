"""Blok D — empat backend anomaly scoring klasik + fusi MINIMUM.

Salinan logika ini WAJIB identik dengan
``experiments/audiax_pipeline_AOT_New_MAC.ipynb`` Bagian 21.

Kenapa empat backend, bukan satu: masing-masing menangkap bentuk penyimpangan
yang berbeda dari memory bank kalibrasi.

===============  =========================================  ===========================
backend          menangkap                                  implementasi
===============  =========================================  ===========================
``cosine``       pergeseran ARAH spektral                   ``1 - max(cos_sim)`` ke bank
``mahalanobis``  pergeseran dengan korelasi antar-dimensi   shrinkage ala Ledoit-Wolf
``knn``          densitas lokal (bank multi-moda)           rata-rata jarak ke k=5 terdekat
``pca``          residu di luar manifold normal             residual rekonstruksi 95% var
===============  =========================================  ===========================

Fusi memakai **MINIMUM** dari z-score tiap backend — bukan rata-rata, bukan
voting, bukan pemilihan backend terbaik otomatis (CLAUDE.md "Keputusan Desain"
#5). Minimum bersifat konservatif: sebuah window baru disebut anomali kalau
SEMUA backend sepakat ia jauh. Konsekuensinya diketahui dan diterima — satu
backend yang berisik cukup untuk menyeret seluruh fusi turun (lihat
``docs/PROGRESS.md`` §"Bug Nyata"). Itu alasan memilih backend dengan hati-hati,
bukan alasan mengganti aturan fusinya.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .config import Config, DEFAULT_CONFIG, SCORING_BACKENDS

__all__ = [
    "BankModel",
    "prepare_bank",
    "compute_all_backend_scores",
    "score_against_prepared_bank",
    "fuse_scores",
    "SCORING_BACKENDS",
]


@dataclass(frozen=True)
class BankModel:
    """Struktur turunan memory bank yang mahal dihitung, dipakai ulang per query.

    Kenapa ada: ``mahalanobis`` butuh inversi kovarians ``D×D`` dan ``pca`` butuh
    SVD. Menghitung ulang keduanya untuk setiap window uji akan membuat inspeksi
    puluhan kali lebih lambat tanpa mengubah hasil sedikit pun — bank-nya sama.
    Nilai numeriknya identik dengan versi naif di notebook.
    """

    bank: np.ndarray                 # [N, D] float64
    bank_normalized: np.ndarray      # [N, D] float64, tiap baris norm 1
    mean: np.ndarray                 # [D]
    inv_cov: np.ndarray              # [D, D] inverse kovarians ter-shrink
    pca_components: np.ndarray       # [k, D] komponen utama
    n: int
    dim: int


def prepare_bank(bank: np.ndarray, cfg: Config = DEFAULT_CONFIG) -> BankModel:
    """Pra-hitung statistik memory bank sekali untuk dipakai banyak query."""
    arr = np.asarray(bank, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("bank harus [N, D], dapat shape " + str(arr.shape))
    n, dim = arr.shape
    if n < 2:
        raise ValueError(
            "Memory bank terlalu kecil (" + str(n) + " embedding); minimal 2 untuk "
            "menghitung kovarians."
        )

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    bank_normalized = arr / np.maximum(norms, 1e-12)

    mu = arr.mean(axis=0)

    # --- kovarians + shrinkage ala Ledoit-Wolf ---
    # Wajib: N (jumlah window kalibrasi, ~100) jauh lebih kecil dari D (256/768),
    # jadi kovarians sampel singular. Menarik sebagian massa ke diagonal
    # (rata-rata varians) membuatnya invertible tanpa membuang struktur korelasi.
    cov = np.cov(arr, rowvar=False)
    cov = np.atleast_2d(cov)
    shrink = cfg.mahalanobis_shrinkage
    cov_shrunk = (1.0 - shrink) * cov + shrink * (np.trace(cov) / dim) * np.eye(dim)
    try:
        inv_cov = np.linalg.inv(cov_shrunk)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov_shrunk)

    # --- basis PCA yang menahan `pca_variance_target` dari varians ---
    centered = arr - mu
    try:
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:  # pragma: no cover - jarang, tapi jangan sampai 500
        _, s, vt = np.linalg.svd(centered + 1e-9 * np.random.default_rng(0).standard_normal(centered.shape), full_matrices=False)
    total = float(np.sum(s ** 2))
    if total < 1e-12:
        components = np.zeros((0, dim), dtype=np.float64)
    else:
        var_ratio = (s ** 2) / total
        cum = np.cumsum(var_ratio)
        k = min(max(1, int(np.searchsorted(cum, cfg.pca_variance_target) + 1)), vt.shape[0])
        components = vt[:k]

    return BankModel(
        bank=arr,
        bank_normalized=bank_normalized,
        mean=mu,
        inv_cov=inv_cov,
        pca_components=components,
        n=n,
        dim=dim,
    )


# --------------------------------------------------------------------------
# Backend individual. Semua mengembalikan skor MENTAH yang "makin besar makin
# anomali"; skala antar-backend tidak sebanding -- itu urusan `fuse_scores`.
# --------------------------------------------------------------------------


def _cosine_score(query: np.ndarray, model: BankModel) -> float:
    """``1 - kemiripan kosinus tertinggi`` terhadap bank (nearest neighbour sudut)."""
    q = query / max(float(np.linalg.norm(query)), 1e-12)
    sims = model.bank_normalized @ q
    return float(1.0 - sims.max())


def _mahalanobis_score(query: np.ndarray, model: BankModel) -> float:
    """Jarak Mahalanobis ke centroid bank, memakai inverse kovarians ter-shrink."""
    diff = query - model.mean
    return float(np.sqrt(max(float(diff @ model.inv_cov @ diff.T), 0.0)))


def _knn_density_score(query: np.ndarray, model: BankModel, k: int) -> float:
    """Rata-rata jarak Euclidean ke ``k`` tetangga terdekat di bank.

    Menangkap kasus yang lolos Mahalanobis: bank yang multi-moda (mis. blower
    punya dua kecepatan) punya "lubang" di tengah yang dekat ke centroid tapi
    jauh dari setiap titik nyata.
    """
    dists = np.linalg.norm(model.bank - query, axis=1)
    k_eff = max(1, min(k, dists.shape[0]))
    return float(np.sort(dists)[:k_eff].mean())


def _pca_residual_score(query: np.ndarray, model: BankModel) -> float:
    """Norma residu setelah proyeksi ke subruang utama bank."""
    if model.pca_components.shape[0] == 0:
        return 0.0
    q_centered = query - model.mean
    comp = model.pca_components
    proj = comp.T @ (comp @ q_centered)
    return float(np.linalg.norm(q_centered - proj))


_BACKEND_FUNCS = {
    "cosine": lambda q, m, cfg: _cosine_score(q, m),
    "mahalanobis": lambda q, m, cfg: _mahalanobis_score(q, m),
    "knn": lambda q, m, cfg: _knn_density_score(q, m, cfg.knn_k),
    "pca": lambda q, m, cfg: _pca_residual_score(q, m),
}


def score_against_prepared_bank(
    query: np.ndarray,
    model: BankModel,
    cfg: Config = DEFAULT_CONFIG,
    backends: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Skor mentah semua backend terhadap bank yang sudah dipra-hitung."""
    q = np.asarray(query, dtype=np.float64).reshape(-1)
    if q.shape[0] != model.dim:
        raise ValueError(
            "Dimensi query " + str(q.shape[0]) + " tidak cocok dengan memory bank "
            "(" + str(model.dim) + "). Baseline dibuat dengan model lain?"
        )
    names = tuple(backends) if backends is not None else tuple(cfg.backends)
    return {name: float(_BACKEND_FUNCS[name](q, model, cfg)) for name in names}


def compute_all_backend_scores(
    query: np.ndarray,
    bank: np.ndarray,
    cfg: Config = DEFAULT_CONFIG,
    backends: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Skor mentah 4 backend untuk satu query terhadap satu memory bank.

    Bentuk pemanggilan yang dipakai notebook. Untuk banyak query terhadap bank
    yang sama, pakai ``prepare_bank()`` + ``score_against_prepared_bank()``:
    hasilnya identik, biayanya jauh lebih murah.
    """
    model = prepare_bank(bank, cfg)
    return score_against_prepared_bank(query, model, cfg, backends)


def fuse_scores(
    raw_scores: Dict[str, float],
    backend_stats: Dict[str, Dict[str, float]],
) -> float:
    """z-normalisasi tiap backend terhadap statistik kalibrasi, lalu ambil MINIMUM.

    ``backend_stats`` berasal dari ``ai.calibration.build_baseline()`` dan berisi
    ``mu``/``sigma`` distribusi self-score leave-one-out mesin itu sendiri. Di
    situlah sifat *instance-specific* sistem ini berada: ambang 3σ berarti "3σ
    relatif terhadap sebaran normal mesin INI", bukan relatif konstanta global.
    """
    if not raw_scores:
        raise ValueError("raw_scores kosong -- tidak ada backend untuk difusikan.")

    z_scores = []
    for name, raw in raw_scores.items():
        stats = backend_stats.get(name)
        if stats is None:
            raise ValueError(
                "backend_stats tidak punya entri untuk backend '" + name + "'. "
                "Baseline dibuat dengan konfigurasi backend yang berbeda?"
            )
        sigma = float(stats["sigma"])
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = 1e-8
        z_scores.append((float(raw) - float(stats["mu"])) / sigma)

    return float(min(z_scores))
