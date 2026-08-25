#!/usr/bin/env python3
"""Penyiap aset AUDIAX — vendor BEATs (source) dan checkpoint model (.pt).

Jalankan SEKALI di host sebelum `docker build` / `docker compose up`:

    python scripts/download_assets.py

Kenapa langkah ini ada. Checkpoint berukuran 361 MB, di atas batas file GitHub
(100 MB) dan terlalu besar untuk masuk git tanpa LFS, jadi ia tidak ada di repo.
Sementara Dockerfile mem-bake isi ``ai/`` ke dalam image supaya container bisa
berjalan tanpa internet sama sekali. Skrip ini yang menjembatani keduanya:
internet dibutuhkan sekali saat menyiapkan, tidak pernah saat menjalankan.

Skrip ini **exit non-zero** kalau ada aset yang belum lengkap. Versi sebelumnya
hanya mencetak peringatan lalu exit 0, sehingga `docker build` tetap jalan dan
menghasilkan image yang rusak — kegagalannya baru muncul sebagai /healthz 503
saat demo.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = REPO_ROOT / "ai" / "vendor" / "beats"
MODELS_DIR = REPO_ROOT / "models"
WEIGHTS_DIR = REPO_ROOT / "ai" / "weights"

BEATS_FILES = {
    "BEATs.py": "https://raw.githubusercontent.com/microsoft/unilm/master/beats/BEATs.py",
    "backbone.py": "https://raw.githubusercontent.com/microsoft/unilm/master/beats/backbone.py",
    "modules.py": "https://raw.githubusercontent.com/microsoft/unilm/master/beats/modules.py",
}

#: URL checkpoint hasil training tim, di-host sebagai aset GitHub Release pada
#: repo yang sama. Sengaja dijadikan default alih-alih wajib disetel manual:
#: orang yang meng-clone repo ini harus bisa menjalankan `docker compose up`
#: tanpa membaca dokumentasi lebih dulu, dan satu env var yang lupa disetel
#: adalah selisih antara demo yang jalan dan demo yang mati.
DEFAULT_CHECKPOINT_URL = (
    "https://github.com/AIC-SahabatOllie/audiax_model/releases/download/"
    "v0.1.0-weights/beats_finetuned.pt"
)

#: Timpa default di atas kalau checkpoint di-host di tempat lain (mirror
#: internal, Hugging Face, atau hasil training sendiri):
#:     export AUDIAX_CHECKPOINT_URL=https://.../beats_finetuned.pt
ENV_CHECKPOINT_URL = "AUDIAX_CHECKPOINT_URL"

#: sha256 aset di rilis v0.1.0-weights. Ukuran minimum saja tidak cukup: file
#: yang terpotong di 60% tetap lolos ambang 50 MB tapi gagal saat torch.load,
#: dan pesan errornya tidak akan menyebut-nyebut unduhan.
EXPECTED_CHECKPOINT_SHA256 = "e3feaefb3099882b7cc234799ee37455f677efe70b2df7525362ec6b5f8ae509"

#: Ukuran minimum yang masuk akal untuk checkpoint BEATs. Unduhan yang terputus
#: atau halaman error HTML yang tersimpan sebagai .pt akan jauh di bawah ini,
#: dan lebih baik ketahuan sekarang daripada sebagai crash saat memuat bobot.
MIN_CHECKPOINT_BYTES = 50 * 1024 * 1024


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


#: Berapa kali unduhan diulang sebelum menyerah. Bukan kemewahan: checkpoint
#: 345 MB lewat koneksi rumahan bisa berjalan puluhan menit, dan satu gangguan
#: sesaat di tengahnya pernah membuat seluruh unduhan mengulang dari nol lalu
#: skripnya menyerah. Di lokasi penjurian, itu berarti demo mati karena WiFi.
DOWNLOAD_ATTEMPTS = 4


def _download(url: str, dest: Path, attempts: int = DOWNLOAD_ATTEMPTS) -> bool:
    """Unduh url ke dest, melanjutkan dari file .part kalau ada.

    Melanjutkan, bukan mengulang: byte yang sudah turun dipertahankan dan
    permintaan berikutnya memakai header ``Range``. Untuk file 345 MB, selisih
    antara "lanjut dari 80%" dan "ulang dari 0%" adalah selisih antara demo yang
    jalan dan demo yang tidak.

    URL asli selalu yang diminta ulang, bukan URL hasil redirect. Aset GitHub
    Release diarahkan ke CDN dengan tanda tangan yang kedaluwarsa; menyimpan
    URL redirect lalu memakainya lagi akan gagal justru pada unduhan panjang
    yang paling butuh diulang.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, attempts + 1):
        resume_from = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url)
        if resume_from:
            req.add_header("Range", f"bytes={resume_from}-")
            print(f"      melanjutkan dari {resume_from / 1024 / 1024:.0f} MB")

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                # Server boleh mengabaikan Range dan mengirim 200 dari awal.
                # Kalau itu terjadi, menimpa dari nol adalah satu-satunya yang
                # benar -- menambahkan ke .part yang sudah ada akan menghasilkan
                # file rusak yang ukurannya justru terlihat wajar.
                restarted = resume_from and resp.status != 206
                if restarted:
                    resume_from = 0
                mode = "ab" if resume_from else "wb"

                with tmp.open(mode) as out:
                    total = resume_from
                    last_report = total
                    while chunk := resp.read(1024 * 256):
                        out.write(chunk)
                        total += len(chunk)
                        if total - last_report >= 32 * 1024 * 1024:
                            print(f"      {total / 1024 / 1024:.0f} MB...")
                            last_report = total
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # .part sengaja TIDAK dihapus -- itulah yang membuat percobaan
            # berikutnya bisa melanjutkan alih-alih mengulang.
            print(f"    [percobaan {attempt}/{attempts} gagal] {exc}", file=sys.stderr)
            if attempt == attempts:
                print(f"    [GAGAL] menyerah setelah {attempts} percobaan", file=sys.stderr)
                return False
            time.sleep(2 * attempt)
            continue

        # Rename hanya setelah unduhan utuh, supaya file separuh tidak pernah
        # terlihat seperti aset yang valid oleh proses lain.
        tmp.replace(dest)
        return True

    return False


def setup_beats_vendor() -> List[str]:
    print(f"[*] Vendor BEATs: {VENDOR_DIR}")
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    problems: List[str] = []
    for filename, url in BEATS_FILES.items():
        dest = VENDOR_DIR / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  - [ADA]  {filename}")
            continue
        print(f"  - [UNDUH] {filename}")
        if not _download(url, dest):
            problems.append(f"vendor BEATs {filename} gagal diunduh dari {url}")
    return problems


def _existing_checkpoint() -> Optional[Path]:
    for candidate in (
        WEIGHTS_DIR / "beats_finetuned.pt",
        WEIGHTS_DIR / "BEATs_iter3_plus_AS20K.pt",
        MODELS_DIR / "BEATs_iter3_plus_AS20K.pt",
    ):
        if candidate.exists() and candidate.stat().st_size >= MIN_CHECKPOINT_BYTES:
            return candidate
    return None


def setup_checkpoint() -> List[str]:
    print("\n[*] Checkpoint BEATs")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    found = _existing_checkpoint()
    if found is not None:
        size_mb = found.stat().st_size / (1024 * 1024)
        print(f"  - [ADA]  {found.relative_to(REPO_ROOT)} ({size_mb:.1f} MB)")
        return []

    url = os.environ.get(ENV_CHECKPOINT_URL, "").strip() or DEFAULT_CHECKPOINT_URL

    dest = WEIGHTS_DIR / "beats_finetuned.pt"
    print(f"  - [UNDUH] {url}")
    if not _download(url, dest):
        return [f"checkpoint gagal diunduh dari {url}"]

    size = dest.stat().st_size
    if size < MIN_CHECKPOINT_BYTES:
        dest.unlink(missing_ok=True)
        return [
            f"checkpoint hasil unduhan cuma {size / 1024 / 1024:.1f} MB — "
            "kemungkinan URL salah atau mengembalikan halaman error, bukan file bobot"
        ]

    # Hash hanya diperiksa untuk URL default, karena hanya aset itu yang
    # hash-nya kita ketahui. Mirror atau checkpoint hasil training sendiri
    # tentu berbeda, dan menolaknya di sini akan salah.
    if url == DEFAULT_CHECKPOINT_URL:
        digest = _sha256(dest)
        if digest != EXPECTED_CHECKPOINT_SHA256:
            dest.unlink(missing_ok=True)
            return [
                "checkpoint terunduh utuh secara ukuran tapi sha256-nya tidak cocok.\n"
                f"      diharapkan: {EXPECTED_CHECKPOINT_SHA256}\n"
                f"      didapat   : {digest}\n"
                "      File dihapus. Jalankan ulang; kalau tetap berbeda, aset di rilis "
                "sudah berubah dan konstanta di skrip ini harus diperbarui."
            ]
        print("    [OK] sha256 cocok")

    print(f"    [OK] {size / 1024 / 1024:.1f} MB")
    return []


def main() -> int:
    problems = setup_beats_vendor() + setup_checkpoint()

    print()
    if problems:
        print("=" * 70, file=sys.stderr)
        print("ASET BELUM LENGKAP — `docker build` akan gagal:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        return 1

    print("Semua aset siap. Lanjut:  docker compose up --build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
