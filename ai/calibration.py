"""Blok C — kalibrasi instance-specific (memory bank + statistik per mesin).

Salinan logika ini WAJIB identik dengan
``experiments/audiax_pipeline_AOT_New_MAC.ipynb`` Bagian 20.

Inti idenya: tidak ada ambang universal untuk "suara blower normal". Yang ada
adalah "normal untuk unit INI, di ruangan INI, dengan HP INI". Karena itu
kalibrasi menghitung distribusi skor **self-score leave-one-out**: untuk tiap
window kalibrasi, skor dihitung terhadap bank yang sudah mengeluarkan window itu
sendiri. Tanpa leave-one-out setiap window akan berjarak nol dari dirinya
sendiri, ``sigma`` runtuh, dan seluruh klip normal dilaporkan WARNING — bug ini
pernah nyata terjadi dan tercatat di ``docs/PROGRESS.md``.

Apa yang BUKAN terjadi di sini (batasan rulebook #1 dan #2): tidak ada gradient
descent, tidak ada fine-tuning, tidak ada auto-tuning berkelanjutan. Yang
dihitung hanyalah statistik deskriptif (mean/std), sekali, lalu dibekukan.

Artefak keluarannya (``MachineBaseline``) disimpan di **klien**, bukan di server
(batasan rulebook #3 — nol state di server, nol database).
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np

from .config import Config, DEFAULT_CONFIG, SCHEMA_VERSION
from .scoring import fuse_scores, prepare_bank, score_against_prepared_bank

__all__ = ["MachineBaseline", "build_baseline", "model_fingerprint"]

_EMBEDDING_DTYPE = np.float16   # cukup untuk kosinus; separuh ukuran payload


def model_fingerprint(
    encoder: Any,
    adapter: Any,
    cfg: Config = DEFAULT_CONFIG,
) -> str:
    """Identitas gabungan (encoder, adapter, parameter yang mengubah embedding).

    16 karakter pertama SHA256 dari triplet itu. Fungsinya sebagai *interlock*:
    sebuah ``MachineBaseline`` hanya sah dipakai oleh kombinasi model+parameter
    yang sama persis dengan saat ia dibuat. Kalau checkpoint diganti atau
    ``enable_denoise`` dinyalakan, memory bank lama tidak lagi hidup di ruang
    yang sama — dan ``decide()`` akan menolaknya dengan status
    ``KALIBRASI_KURANG`` alih-alih mengeluarkan z-score yang tidak berarti.

    Encoder/adapter tiruan di test cukup mengekspos atribut ``fingerprint``;
    kalau tidak ada, nama kelasnya yang dipakai.
    """
    enc_id = str(getattr(encoder, "fingerprint", type(encoder).__name__))
    ada_id = str(getattr(adapter, "fingerprint", type(adapter).__name__))
    payload = json.dumps(
        {
            "encoder": enc_id,
            "adapter": ada_id,
            "cfg": {k: v for k, v in cfg.fingerprint_fields()},
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class MachineBaseline:
    """Baseline satu unit mesin — memory bank + skala skor miliknya sendiri.

    Attributes:
        schema_version: Versi format JSON; ketidakcocokan ditolak saat dibaca.
        machine_label: Label bebas dari operator ("Blower Oven 1").
        created_at: Waktu kalibrasi (ISO 8601, UTC).
        model_fingerprint: Lihat :func:`model_fingerprint`.
        n_windows: Jumlah window yang membentuk bank.
        embeddings: ``[N, D]`` memory bank, float32 di memori.
        backend_stats: ``{"cosine": {"mu": .., "sigma": ..}, ...}``.
        calibration_quality: ``"baik"`` atau ``"rendah"``.
    """

    schema_version: str
    machine_label: str
    created_at: str
    model_fingerprint: str
    n_windows: int
    embeddings: np.ndarray
    backend_stats: Dict[str, Dict[str, float]]
    calibration_quality: str
    notes: Dict[str, Any] = field(default_factory=dict)

    # ---------------- serialisasi ----------------

    def to_json_dict(self) -> Dict[str, Any]:
        """Bentuk yang aman dikirim lewat HTTP dan disimpan klien.

        Embeddings dikompres ke float16 lalu base64. Presisi float16 (~3 digit
        desimal) jauh di atas yang dibutuhkan skor kosinus, sementara ukurannya
        separuh float32 — sekitar 85 KB untuk 119 window × 768 dimensi, cukup
        kecil untuk IndexedDB di browser HP.
        """
        arr = np.ascontiguousarray(self.embeddings, dtype=_EMBEDDING_DTYPE)
        return {
            "schema_version": self.schema_version,
            "machine_label": self.machine_label,
            "created_at": self.created_at,
            "model_fingerprint": self.model_fingerprint,
            "n_windows": int(self.n_windows),
            "embedding_shape": list(arr.shape),
            "embedding_dtype": str(arr.dtype),
            "embeddings_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
            "backend_stats": {
                name: {"mu": float(s["mu"]), "sigma": float(s["sigma"])}
                for name, s in self.backend_stats.items()
            },
            "calibration_quality": self.calibration_quality,
            "notes": dict(self.notes),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), separators=(",", ":"))

    @classmethod
    def from_json_dict(cls, d: Dict[str, Any]) -> "MachineBaseline":
        """Kebalikan :meth:`to_json_dict`, dengan validasi yang galak.

        Baseline datang dari klien, jadi isinya tidak tepercaya. Setiap
        ketidakcocokan dilaporkan sebagai ``ValueError`` supaya ``service/``
        bisa memetakannya ke HTTP 400, bukan 500.
        """
        if not isinstance(d, dict):
            raise ValueError("baseline harus berupa objek JSON.")

        required = (
            "schema_version",
            "machine_label",
            "model_fingerprint",
            "n_windows",
            "embedding_shape",
            "embeddings_b64",
            "backend_stats",
            "calibration_quality",
        )
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError("baseline tidak lengkap, field hilang: " + ", ".join(missing))

        if str(d["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(
                "Versi skema baseline " + str(d["schema_version"]) + " tidak cocok dengan "
                "versi yang didukung " + SCHEMA_VERSION + ". Lakukan kalibrasi ulang."
            )

        shape = tuple(int(v) for v in d["embedding_shape"])
        if len(shape) != 2:
            raise ValueError("embedding_shape harus [N, D], dapat " + str(list(shape)))

        dtype = np.dtype(d.get("embedding_dtype", str(_EMBEDDING_DTYPE().dtype)))
        try:
            raw = base64.b64decode(d["embeddings_b64"], validate=True)
        except Exception as exc:
            raise ValueError("embeddings_b64 bukan base64 yang sah.") from exc

        expected_bytes = int(np.prod(shape)) * dtype.itemsize
        if len(raw) != expected_bytes:
            raise ValueError(
                "Ukuran embeddings tidak konsisten: " + str(len(raw)) + " byte, "
                "seharusnya " + str(expected_bytes) + " untuk shape " + str(list(shape)) + "."
            )

        embeddings = np.frombuffer(raw, dtype=dtype).reshape(shape).astype(np.float32)

        stats_in = d["backend_stats"]
        if not isinstance(stats_in, dict) or not stats_in:
            raise ValueError("backend_stats kosong atau bukan objek.")
        backend_stats: Dict[str, Dict[str, float]] = {}
        for name, s in stats_in.items():
            if not isinstance(s, dict) or "mu" not in s or "sigma" not in s:
                raise ValueError("backend_stats['" + str(name) + "'] harus punya mu dan sigma.")
            backend_stats[str(name)] = {"mu": float(s["mu"]), "sigma": float(s["sigma"])}

        quality = str(d["calibration_quality"])
        if quality not in ("baik", "rendah"):
            raise ValueError("calibration_quality tidak dikenal: " + quality)

        return cls(
            schema_version=str(d["schema_version"]),
            machine_label=str(d["machine_label"]),
            created_at=str(d.get("created_at", "")),
            model_fingerprint=str(d["model_fingerprint"]),
            n_windows=int(d["n_windows"]),
            embeddings=embeddings,
            backend_stats=backend_stats,
            calibration_quality=quality,
            notes=dict(d.get("notes") or {}),
        )

    @classmethod
    def from_json(cls, s: str) -> "MachineBaseline":
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError("baseline_json bukan JSON yang sah: " + str(exc)) from exc
        return cls.from_json_dict(parsed)


def build_baseline(
    embeddings: np.ndarray,
    machine_label: str,
    model_fingerprint: str,
    cfg: Config = DEFAULT_CONFIG,
) -> MachineBaseline:
    """Bangun ``MachineBaseline`` dari embedding SELURUH window kalibrasi.

    Args:
        embeddings: ``[N, D]`` hasil ``adapter.project()`` atas window kalibrasi.
        machine_label: Label unit dari operator.
        model_fingerprint: Hasil :func:`model_fingerprint` untuk model yang
            menghasilkan ``embeddings`` — dibawa serta supaya baseline tidak
            pernah dipakai lintas model.
        cfg: Parameter statis.

    Alur: untuk tiap embedding, hitung skor 4 backend terhadap bank
    **leave-one-out**, kumpulkan distribusinya per backend, simpan mean/std.
    """
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("embeddings harus [N, D], dapat shape " + str(arr.shape))

    n_total = arr.shape[0]
    if n_total < 2:
        raise ValueError(
            "Butuh minimal 2 window kalibrasi untuk membentuk memory bank, dapat "
            + str(n_total) + "."
        )

    # Batasi ukuran bank: menjaga payload klien dan biaya scoring tetap terikat.
    # Sub-sampling merata (bukan potong 240 pertama) supaya seluruh durasi
    # rekaman tetap terwakili, termasuk variasi di menit terakhir.
    truncated = False
    if n_total > cfg.max_calibration_windows:
        idx = np.linspace(0, n_total - 1, cfg.max_calibration_windows).round().astype(int)
        arr = arr[np.unique(idx)]
        truncated = True

    n = arr.shape[0]
    quality = "baik" if n >= cfg.min_calibration_windows else "rendah"

    backend_names = tuple(cfg.backends)
    raw_by_backend: Dict[str, list] = {name: [] for name in backend_names}

    for i in range(n):
        bank_loo = np.delete(arr, i, axis=0)
        if bank_loo.shape[0] < 2:
            continue
        model = prepare_bank(bank_loo, cfg)
        raw = score_against_prepared_bank(arr[i], model, cfg, backend_names)
        for name, val in raw.items():
            raw_by_backend[name].append(val)

    backend_stats: Dict[str, Dict[str, float]] = {}
    for name in backend_names:
        scores = raw_by_backend[name]
        values = np.asarray(scores, dtype=np.float64) if scores else np.array([0.0])
        # +1e-8 mencegah pembagian nol saat seluruh window kalibrasi identik
        # (mis. rekaman uji sintetis); sama persis dengan notebook Bagian 20.
        backend_stats[name] = {
            "mu": float(values.mean()),
            "sigma": float(values.std() + 1e-8),
        }

    return MachineBaseline(
        schema_version=SCHEMA_VERSION,
        machine_label=machine_label,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_fingerprint=model_fingerprint,
        n_windows=int(n),
        embeddings=arr.astype(np.float32),
        backend_stats=backend_stats,
        calibration_quality=quality,
        notes={
            "backends": list(backend_names),
            "embedding_dim": int(arr.shape[1]),
            "windows_discovered": int(n_total),
            "subsampled": bool(truncated),
            "min_calibration_windows": int(cfg.min_calibration_windows),
        },
    )


def self_score_distribution(
    baseline: MachineBaseline,
    cfg: Config = DEFAULT_CONFIG,
) -> np.ndarray:
    """z hasil fusi untuk tiap window kalibrasi (leave-one-out).

    Alat inspeksi, bukan bagian jalur inferensi: berguna untuk memeriksa bahwa
    ambang ``z_warning``/``z_critical`` benar-benar duduk di ekor distribusi
    normal mesin ini, bukan di tengahnya.
    """
    arr = np.asarray(baseline.embeddings, dtype=np.float32)
    out = []
    for i in range(arr.shape[0]):
        bank_loo = np.delete(arr, i, axis=0)
        if bank_loo.shape[0] < 2:
            continue
        model = prepare_bank(bank_loo, cfg)
        raw = score_against_prepared_bank(arr[i], model, cfg, tuple(baseline.backend_stats))
        out.append(fuse_scores(raw, baseline.backend_stats))
    return np.asarray(out, dtype=np.float64)
