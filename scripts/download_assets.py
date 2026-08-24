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

import os
import sys
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

#: URL checkpoint hasil training tim. Tidak di-hardcode karena setiap tim
#: meng-host-nya sendiri (GitHub Release, Hugging Face, atau storage lain).
#: Setel sebelum menjalankan skrip:
#:     export AUDIAX_CHECKPOINT_URL=https://github.com/<org>/<repo>/releases/download/<tag>/beats_finetuned.pt
ENV_CHECKPOINT_URL = "AUDIAX_CHECKPOINT_URL"

#: Ukuran minimum yang masuk akal untuk checkpoint BEATs. Unduhan yang terputus
#: atau halaman error HTML yang tersimpan sebagai .pt akan jauh di bawah ini,
#: dan lebih baik ketahuan sekarang daripada sebagai crash saat memuat bobot.
MIN_CHECKPOINT_BYTES = 50 * 1024 * 1024


def _download(url: str, dest: Path) -> bool:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
            total = 0
            while chunk := resp.read(1024 * 256):
                out.write(chunk)
                total += len(chunk)
                if total % (32 * 1024 * 1024) < 1024 * 256:
                    print(f"      {total / 1024 / 1024:.0f} MB...")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"    [GAGAL] {exc}", file=sys.stderr)
        return False
    # Rename hanya setelah unduhan utuh, supaya file separuh tidak pernah
    # terlihat seperti aset yang valid oleh proses lain.
    tmp.replace(dest)
    return True


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

    url = os.environ.get(ENV_CHECKPOINT_URL, "").strip()
    if not url:
        return [
            "checkpoint BEATs belum ada dan "
            + ENV_CHECKPOINT_URL
            + " belum disetel.\n"
            "      Setel ke URL rilis checkpoint tim, lalu jalankan ulang:\n"
            "        export "
            + ENV_CHECKPOINT_URL
            + "=https://github.com/<org>/<repo>/releases/download/<tag>/beats_finetuned.pt\n"
            "      Atau salin manual file .pt ke ai/weights/beats_finetuned.pt"
        ]

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
