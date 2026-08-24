"""Fase 3 — LoRA fine-tune Gemma 3 270M untuk lapisan advisory AUDIAX.

Apa yang model ini HARUS pelajari, dan apa yang TIDAK.

Tidak perlu dipelajari: bentuk JSON. Saat produksi, llama.cpp menegakkannya
lewat grammar GBNF (``response_format: json_schema``), jadi keluaran yang
melanggar skema secara struktural tidak mungkin terjadi. Fase 0 mengukur ini:
tanpa grammar 0/10 valid, dengan grammar 100%.

Yang perlu dipelajari adalah PERILAKU ISI -- tiga hal yang gagal pada model
dasar di Fase 0:

  1. Jangan membantah keputusan sistem. Model dasar menjawab "mesin masih aman
     dipakai" pada status WARNING yang gerbang keselamatannya menyuruh mematikan.
  2. Jangan mengarang. Model dasar menjelaskan crest factor sebagai "indikator
     kualitas kalibrasi, makin tinggi makin baik" -- salah, dan terbalik.
  3. Eskalasi saat bahaya. Operator melapor bau gosong, model dasar menjawab
     dengan checklist biasa tanpa menyuruh berhenti.

Menyempitkan beban belajar ke tiga hal inilah yang membuat model 270M layak
dicoba sama sekali.

Training di CPU. Untuk 270M dengan ~600 contoh pendek itu wajar (~20-40 menit),
dan menghindari unduhan wheel CUDA 2,5 GB yang nyaris menghabiskan disk.

Pakai:
    python train_lora.py --epochs 3
    python train_lora.py --epochs 1 --limit 40      # asap, cek pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE.parent.parent / ".hf_cache"))

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

BASE_MODEL = "unsloth/gemma-3-270m-it"
CORPUS = HERE / "corpus"
OUT_ADAPTER = HERE / "adapter"
OUT_MERGED = HERE / "merged"

# Identik dengan SYSTEM_HINT di bench_phase0.py. Kalau berbeda, model dilatih
# pada konteks yang bukan konteks produksinya.
SYSTEM = (
    "Kamu asisten pemeliharaan mesin blower untuk operator UMKM. "
    "Jawab dalam Bahasa Indonesia sederhana. "
    "Jangan pernah menyebut angka yang tidak ada di FAKTA atau KEPUTUSAN SISTEM. "
    "Jangan menyebut jenis kerusakan. "
    'Balas HANYA JSON dengan kunci: "jawaban", "langkah_berikutnya", '
    '"perlu_teknisi" (boolean), "eskalasi" (boolean).'
)

MAX_LEN = 1024


class AdvisoryDataset(Dataset):
    """Contoh training dengan label prompt di-mask.

    Loss hanya dihitung pada token jawaban. Tanpa masking, model menghabiskan
    kapasitasnya untuk menghafal blok FAKTA/KEPUTUSAN yang panjang dan selalu
    berubah -- padahal blok itu adalah INPUT yang selalu diberikan, bukan sesuatu
    yang perlu diprediksi. Pada model 270M, kapasitas yang terbuang itu mahal.
    """

    def __init__(self, rows: List[Dict[str, Any]], tok) -> None:
        self.items: List[Dict[str, List[int]]] = []
        skipped = 0
        for r in rows:
            # Gemma tidak punya peran "system" terpisah; templatenya menggabungkan
            # instruksi sistem ke giliran user pertama. Digabung eksplisit di sini
            # supaya bentuknya sama persis dengan yang dikirim llama-server.
            user = SYSTEM + "\n\n" + r["prompt"]
            # transformers 5.x mengembalikan BatchEncoding di sini, 4.x list[int].
            # Ditangani dua-duanya supaya skrip ini tidak pecah saat versi berubah.
            enc = tok.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=True,
                add_generation_prompt=True,
            )
            prompt_ids = enc["input_ids"] if hasattr(enc, "keys") else enc
            if prompt_ids and isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]
            prompt_ids = list(prompt_ids)
            answer_ids = tok(r["completion"] + tok.eos_token, add_special_tokens=False)["input_ids"]

            ids = prompt_ids + answer_ids
            if len(ids) > MAX_LEN:
                skipped += 1
                continue
            labels = [-100] * len(prompt_ids) + answer_ids
            self.items.append({"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)})
        if skipped:
            print(f"  {skipped} contoh dilewati karena melebihi {MAX_LEN} token")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, List[int]]:
        return self.items[i]


def load_split(name: str) -> List[Dict[str, Any]]:
    path = CORPUS / f"{name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="potong dataset, untuk uji asap")
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 8)

    print(f"memuat {BASE_MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    model.config.use_cache = False

    train_rows = load_split("train")
    val_rows = load_split("val")
    if args.limit:
        train_rows, val_rows = train_rows[: args.limit], val_rows[: max(2, args.limit // 8)]
    if not train_rows:
        print("corpus/train.jsonl kosong -- jalankan gen_corpus.py dulu")
        return 2

    print(f"train {len(train_rows)} | val {len(val_rows)}", flush=True)
    train_ds = AdvisoryDataset(train_rows, tok)
    val_ds = AdvisoryDataset(val_rows, tok) if val_rows else None

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"parameter dilatih: {trainable/1e6:.1f}M dari {total/1e6:.0f}M ({trainable/total*100:.2f}%)", flush=True)

    targs = TrainingArguments(
        output_dir=str(HERE / "runs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        # transformers 5.x membuang warmup_ratio; hanya warmup_steps yang ada.
        warmup_steps=10,
        logging_steps=5,
        # Checkpoint tiap 20 langkah. Training penuh berjalan ~3 jam di CPU dan
        # job panjang di lingkungan ini sudah beberapa kali dihentikan di tengah;
        # tanpa ini seluruh jam yang sudah berjalan hangus. save_total_limit
        # menjaga disk tidak dipenuhi checkpoint lama.
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        eval_strategy="epoch" if val_ds else "no",
        report_to=[],
        use_cpu=True,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )

    # Lanjut dari checkpoint terakhir kalau ada, supaya penghentian di tengah
    # tidak berarti mengulang dari nol.
    runs = HERE / "runs"
    ckpts = sorted(runs.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])) if runs.exists() else []
    resume = str(ckpts[-1]) if ckpts else None
    if resume:
        print(f"melanjutkan dari {ckpts[-1].name}", flush=True)

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    print(f"\nselesai dalam {(time.time()-t0)/60:.1f} menit", flush=True)

    OUT_ADAPTER.mkdir(exist_ok=True)
    model.save_pretrained(str(OUT_ADAPTER))
    tok.save_pretrained(str(OUT_ADAPTER))
    print(f"adapter -> {OUT_ADAPTER}")

    # Merge supaya bisa dikonversi ke GGUF: llama.cpp memuat satu set bobot,
    # bukan basis + adapter terpisah.
    merged = model.merge_and_unload()
    OUT_MERGED.mkdir(exist_ok=True)
    merged.save_pretrained(str(OUT_MERGED), safe_serialization=True)
    tok.save_pretrained(str(OUT_MERGED))
    print(f"merged  -> {OUT_MERGED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
