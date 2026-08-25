"""Hasilkan golden file prompt untuk test anti-skew di repo backend.

Kenapa ini ada sebagai skrip, bukan file yang diketik tangan:

Golden file pertama di repo backend ditulis manual dan salah di tiga tempat --
`urgensi` memakai spasi padahal korpus memakai garis bawah, checklist terpotong
2 dari 4 butir, dan `eskalasi_bila` diringkas. Test yang membandingkan Go ke
golden yang salah lebih buruk daripada tidak ada test sama sekali: ia akan
memaksa Go menyesuaikan diri ke format yang model tidak pernah lihat.

Karena itu golden HARUS dirender oleh `gen_corpus.py::render_prompt` -- fungsi
yang persis sama, tidak cuma mirip, dengan yang menghasilkan tiap contoh
training. Kalau golden dan korpus berasal dari satu fungsi, mustahil keduanya
berbeda.

Pakai:
    python gen_golden.py                    # tulis ke repo backend
    python gen_golden.py --print            # cetak saja, jangan tulis
    python gen_golden.py --backend PATH     # kalau repo backend bukan tetangga
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen_corpus as gc  # noqa: E402  (butuh sys.path di atas)

DEFAULT_BACKEND = HERE.parent.parent.parent / "audiax_backend"

# Skenario yang dipilih bukan sembarangan: WARNING dengan gerbang keselamatan
# aktif, z terisi, riwayat KOSONG. Riwayat kosong-lah yang memicu penghapusan
# blok `RIWAYAT:` sekaligus penggabungan dua baris kosong berurutan -- satu-
# satunya tempat `collapseBlankRuns()` di Go benar-benar berbeda perilakunya
# dari penghapusan baris biasa, dan karenanya satu-satunya tempat kedua
# implementasi paling mungkin diam-diam menyimpang.
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "golden_prompt_no_history.txt": {
        "ctx": {
            "status": "WARNING",
            "z": 3.4,
            "dominant_indicator": "crest_factor",
            "drive_type": "belt",
            "recency": ">6bln",
            "machine_age": "3-5th",
            "hours_per_day": ">8",
            "has_backup": False,
            "load_state": "bermuatan",
        },
        "question": "sabuknya udah kenceng kok, tapi tadi ada bau gosong dikit",
        "history": [],
    },
}


def render(backend: Path, spec: Dict[str, Any]) -> str:
    template = (backend / "internal/advisory/prompt_template.txt").read_text(encoding="utf-8")
    raw = json.loads((backend / "internal/advisory/decision_table.json").read_text(encoding="utf-8"))
    cells: List[Dict[str, Any]] = raw["cells"] if isinstance(raw, dict) else raw

    cell = gc.lookup(cells, spec["ctx"])
    if cell is None:
        raise SystemExit(f"tidak ada sel keputusan untuk konteks {spec['ctx']}")

    facts = gc.build_facts(spec["ctx"], cell, spec["question"], spec["history"])
    return gc.render_prompt(template, facts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", type=Path, default=DEFAULT_BACKEND)
    ap.add_argument("--print", dest="only_print", action="store_true")
    args = ap.parse_args()

    backend = args.backend.resolve()
    if not (backend / "internal/advisory/prompt_template.txt").exists():
        raise SystemExit(f"repo backend tidak ditemukan di {backend} -- pakai --backend")

    outdir = backend / "internal/advisory/testdata"
    outdir.mkdir(parents=True, exist_ok=True)

    for name, spec in SCENARIOS.items():
        text = render(backend, spec)
        if args.only_print:
            sys.stdout.write(text)
            continue
        # newline="\n" wajib: di Windows tulisan default jadi CRLF, dan walau
        # test Go menormalkannya, file yang ter-commit jadi berbeda tiap OS.
        (outdir / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"-> {outdir / name}  ({len(text.encode('utf-8'))} byte)")

    if not args.only_print:
        print("\nJalankan di repo backend untuk memverifikasi:")
        print("  go test ./internal/advisory/ -run MatchesPythonGolden -v")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
