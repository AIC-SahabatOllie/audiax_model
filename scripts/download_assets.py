#!/usr/bin/env python3
"""
Audiax MLOps Asset Downloader
=============================
Mengunduh dependensi vendor BEATs (source code) dan memvalidasi keberadaan bobot model (.pt).
"""

import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = REPO_ROOT / "ai" / "vendor" / "beats"
MODELS_DIR = REPO_ROOT / "models"
WEIGHTS_DIR = REPO_ROOT / "ai" / "weights"

BEATS_FILES = {
    "BEATs.py": "https://raw.githubusercontent.com/microsoft/unilm/master/beats/BEATs.py",
    "backbone.py": "https://raw.githubusercontent.com/microsoft/unilm/master/beats/backbone.py",
    "modules.py": "https://raw.githubusercontent.com/microsoft/unilm/master/beats/modules.py",
}

def setup_beats_vendor():
    print(f"[*] Menyiapkan vendor BEATs di: {VENDOR_DIR}")
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    
    for filename, url in BEATS_FILES.items():
        dest = VENDOR_DIR / filename
        if not dest.exists() or dest.stat().st_size == 0:
            print(f"  - Mengunduh {filename} dari GitHub...")
            try:
                urllib.request.urlretrieve(url, dest)
                print(f"    [OK] {filename} berhasil diunduh.")
            except Exception as e:
                print(f"    [FAIL] Gagal mengunduh {filename}: {e}", file=sys.stderr)
        else:
            print(f"  - [EXISTS] {filename} sudah tersedia.")

def check_model_weights():
    print("\n[*] Memeriksa ketersediaan bobot checkpoint (.pt)...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    
    candidates = [
        WEIGHTS_DIR / "beats_finetuned.pt",
        WEIGHTS_DIR / "BEATs_iter3_plus_AS20K.pt",
        MODELS_DIR / "BEATs_iter3_plus_AS20K.pt",
    ]
    
    found = [c for c in candidates if c.exists()]
    if found:
        print("  [OK] Ditemukan checkpoint model:")
        for f in found:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    - {f.relative_to(REPO_ROOT)} ({size_mb:.1f} MB)")
    else:
        print("  [!] PERINGATAN: Belum ditemukan file checkpoint model (.pt).")
        print("      Silakan letakkan salah satu file bobot di:")
        print(f"      1. {WEIGHTS_DIR / 'beats_finetuned.pt'} (Hasil training MAC)")
        print(f"      2. {MODELS_DIR / 'BEATs_iter3_plus_AS20K.pt'} (Pretrained raw)")
        print("      Lihat models/README.md untuk tautan unduhan resmi.")

if __name__ == "__main__":
    setup_beats_vendor()
    check_model_weights()
