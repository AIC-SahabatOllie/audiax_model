"""Gabungkan adapter LoRA dari sebuah checkpoint jadi bobot utuh.

Gunanya: mendapat sinyal LEBIH AWAL. Training penuh berjalan ~3,5 jam, tapi
checkpoint tersimpan tiap 20 langkah. Dengan menggabungkan checkpoint di akhir
epoch pertama (~1 jam), gerbang go/no-go bisa dijawab lebih cepat -- dan kalau
tidak ada perbaikan sama sekali pada perilaku keselamatan, dua setengah jam
sisanya tidak perlu dibakar.

Pakai:
    python merge_ckpt.py                      # checkpoint terbaru -> merged_ckpt/
    python merge_ckpt.py --step 60 --out m60
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE.parent.parent / ".hf_cache"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from train_lora import BASE_MODEL


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=0, help="0 = checkpoint terbaru")
    ap.add_argument("--out", type=str, default="merged_ckpt")
    args = ap.parse_args()

    runs = HERE / "runs"
    ckpts = sorted(runs.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not ckpts:
        print("belum ada checkpoint di runs/")
        return 1
    ck = next((c for c in ckpts if c.name == f"checkpoint-{args.step}"), ckpts[-1]) if args.step else ckpts[-1]
    print(f"menggabungkan {ck.name} ...", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(ck))
    merged = model.merge_and_unload()

    # Gemma 3 mengirim tokenizer dengan 262145 token tapi matriks embedding
    # 262144 baris -- token terakhir tidak dipakai di varian teks-saja. Konverter
    # GGUF menolak selisih itu (`max(vocab) < vocab_size` gagal tepat di batas),
    # dan menaikkan vocab_size di config saja justru membuat metadata berbohong:
    # llama.cpp lalu menolak dengan "expected 262145, got 262144".
    # Yang benar adalah memadankan bobotnya, bukan angkanya.
    if merged.get_input_embeddings().weight.shape[0] < len(tok):
        old = merged.get_input_embeddings().weight.shape[0]
        merged.resize_token_embeddings(len(tok))
        print(f"embedding {old} -> {len(tok)} agar cocok dengan tokenizer")

    out = HERE / args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    merged.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
