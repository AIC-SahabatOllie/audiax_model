# Artefak model advisory

`audiax-advisor-q4km.gguf` — Gemma 3 270M hasil LoRA fine-tune tim, dikuantisasi
Q4_K_M. **249 MB**, cukup kecil untuk di-bake ke image Docker.

| | |
|---|---|
| Base | `unsloth/gemma-3-270m-it` |
| Metode | LoRA r=16 alpha=32, 7 proyeksi, 3,8M dari 272M parameter (1,40%) |
| Korpus | 657 giliran, disusun Gemini lalu disaring 7 filter + review manusia |
| Checkpoint | langkah 60 dari 195 (satu epoch), loss 0,66 |
| sha256 | `2108fbd9f9f2219a6fdbb5bc4a6be4bf...` |

Angka evaluasinya ada di `../results.md`. File `.gguf` sendiri **tidak masuk
git** (lihat `.gitignore`) — hasilkan ulang dengan:

```bash
python merge_ckpt.py --step 60 --out merged_e60
python <llama.cpp>/convert_hf_to_gguf.py merged_e60 --outfile adv-f16.gguf --outtype f16
llama-quantize adv-f16.gguf audiax-advisor-q4km.gguf Q4_K_M
```

## Menjalankannya

```bash
llama-server -m audiax-advisor-q4km.gguf -t 4 --port 8081 -c 2048
```

Panggil dengan `response_format: {"type":"json_schema", ...}` supaya grammar
GBNF menjamin bentuk keluaran. Tanpa itu, validitas JSON turun drastis —
lihat Fase 0 di `../results.md`.
