"""CLI demo — jalankan pipeline penuh tanpa server HTTP.

    python -m ai.demo --calibrate kalibrasi.wav --label "Blower Oven 1" \
                      --baseline-out baseline.json
    python -m ai.demo --inspect uji.wav --baseline-in baseline.json
    python -m ai.demo --info

Berguna untuk dua hal: memverifikasi checkpoint benar-benar bisa dimuat di mesin
target sebelum menyalakan ``service/``, dan mendemonstrasikan bahwa ``ai/``
sungguh-sungguh berdiri sendiri tanpa framework web apa pun.

Ini **bukan** skrip pengujian massal (batasan rulebook #4): satu file masuk,
satu kartu keluar, tanpa loop dataset dan tanpa penulisan metrik.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .calibration import MachineBaseline
from .config import DEFAULT_CONFIG, Config


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ai.demo",
        description="Demo AUDIAX tanpa server: kalibrasi dan/atau inspeksi satu file WAV.",
    )
    p.add_argument("--calibrate", metavar="WAV", help="rekaman kalibrasi (~2 menit, kondisi sehat)")
    p.add_argument("--inspect", metavar="WAV", help="rekaman uji (~10 detik)")
    p.add_argument("--label", default="Blower Oven 1", help="label unit mesin untuk kalibrasi")
    p.add_argument("--baseline-out", metavar="JSON", help="tulis baseline hasil kalibrasi ke file")
    p.add_argument("--baseline-in", metavar="JSON", help="baca baseline untuk inspeksi")
    p.add_argument("--device", default=DEFAULT_CONFIG.device, help="cpu | cuda")
    p.add_argument("--denoise", action="store_true", help="nyalakan HPSS (default OFF, lihat CLAUDE.md #4)")
    p.add_argument("--info", action="store_true", help="cetak info model lalu keluar")
    return p


def _print_card(card) -> None:
    d = card.to_dict()
    print("=" * 64)
    print("  STATUS           : " + d["status"])
    print("  z-score          : " + ("n/a" if d["z_score"] is None else format(d["z_score"], ".3f")))
    print("  health score     : " + ("n/a" if d["health_score"] is None else format(d["health_score"], ".1f") + " / 100"))
    print("  kualitas kalibrasi: " + d["calibration_quality"])
    print("  indikator dominan : " + str(d["dominant_indicator"]))
    if d["reason"]:
        print("  catatan          : " + d["reason"])
    print("-" * 64)
    print("  " + d["disclaimer"])
    print("=" * 64)


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)

    from dataclasses import replace

    import ai  # impor di dalam fungsi supaya --help tetap instan

    cfg: Config = replace(DEFAULT_CONFIG, device=args.device, enable_denoise=args.denoise)

    if args.info:
        print(json.dumps(ai.model_info(cfg), indent=2, ensure_ascii=False))
        return 0

    if not args.calibrate and not args.inspect:
        _build_parser().print_help()
        return 2

    baseline: Optional[MachineBaseline] = None

    if args.calibrate:
        print("Kalibrasi dari " + args.calibrate + " ...")
        baseline = ai.calibrate(args.calibrate, machine_label=args.label, cfg=cfg)
        print(
            "  baseline: " + str(baseline.n_windows) + " window, kualitas="
            + baseline.calibration_quality + ", fingerprint=" + baseline.model_fingerprint
        )
        for name, s in baseline.backend_stats.items():
            print("    " + name.ljust(12) + " mu=" + format(s["mu"], ".4f") + "  sigma=" + format(s["sigma"], ".4f"))
        if args.baseline_out:
            Path(args.baseline_out).write_text(baseline.to_json(), encoding="utf-8")
            print("  baseline ditulis ke " + args.baseline_out)

    if args.inspect:
        if baseline is None:
            if not args.baseline_in:
                print("ERROR: --inspect butuh --baseline-in atau dijalankan bersama --calibrate.", file=sys.stderr)
                return 2
            baseline = MachineBaseline.from_json(Path(args.baseline_in).read_text(encoding="utf-8"))
        print("Inspeksi " + args.inspect + " terhadap baseline '" + baseline.machine_label + "' ...")
        _print_card(ai.inspect(args.inspect, baseline=baseline, cfg=cfg))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
