# PROGRESS — AUDIAX (repo AI/model)

**Snapshot per 2026-08-25.** Ini kondisi repo saat ini, bukan log kumulatif.

---

## 1. Ringkasan kondisi

| Komponen | Status |
|---|---|
| `ai/` Blok A–E | ✅ Lengkap, berjalan di container |
| `service/` (FastAPI) | ✅ `/healthz`, `/v1/calibrate`, `/v1/inspect` |
| `tests/` | ✅ boundary, pipeline, service routes |
| Docker / demo | ✅ **Terbukti ujung-ke-ujung** (lihat §2) |
| Distribusi checkpoint | ❌ **Belum** — satu-satunya penghalang demo tersisa |
| Lapisan advisory (Teknisi Saku) | 🔄 Korpus & training jalan, gerbang go/no-go belum dilewati |
| Angka AUC dari notebook | ❌ Belum ada `experiments/results.md` |

---

## 2. Jalur demo — terverifikasi, bukan diasumsikan

Dijalankan pada 2026-08-25, hasil nyata:

```
docker compose build   -> audiax-ai:latest, 2.91 GB
                          verifikasi in-image: beats_finetuned.pt 361.496.686 byte ADA
docker compose up -d   -> container sehat
GET  /healthz          -> {"status":"ok"}      (warm_up sukses = model termuat dari image)
POST /v1/calibrate     -> HTTP 200 dalam 3,2 s
                          embedding_shape [9, 768], 4 backend, quality "rendah"
                          ("rendah" BENAR: sampel 10 detik = 9 window < minimum 20)
```

**Tanpa satu pun volume mount.** Image mandiri sepenuhnya.

### Empat bug packaging yang diperbaiki

1. **`.gitignore` memuat `*.md` dan `/docs`** — setiap dokumentasi baru diabaikan
   diam-diam, termasuk file ini. `README.md` lolos hanya karena sudah ter-track
   lebih dulu. Ini yang paling parah: dokumentasi adalah deliverable yang dinilai.
2. **Volume mount menutupi bobot yang sudah ter-bake.** `COPY ai/ /app/ai/` sudah
   membawa checkpoint ke image, tapi `docker-compose.yml` me-mount `./ai/weights`
   di atasnya. Pada clone bersih folder itu kosong → mount menang → `warm_up()`
   gagal → `/healthz` 503 selamanya. Mount dipindah ke `docker-compose.dev.yml`.
3. **Build lolos walau aset tidak ada** → image rusak yang gagal saat runtime, di
   depan juri. Sekarang build gagal lebih dulu dengan instruksi perbaikannya.
4. **`download_assets.py` cuma memeriksa lalu exit 0.** Kini benar-benar
   mengunduh, memvalidasi ukuran minimum, dan exit non-zero kalau kurang.

### ⚠️ Sisa penghalang demo

`beats_finetuned.pt` (345 MB) belum di-host di mana pun. Orang yang meng-clone
repo tidak punya cara mendapatkannya. **Unggah ke GitHub Release**, lalu
`AUDIAX_CHECKPOINT_URL` di README berfungsi. Sampai itu dikerjakan, jalur "juri
clone lalu jalankan" masih putus.

---

## 3. Lapisan advisory — Teknisi Saku

Fitur pendamping: menjelaskan `HealthCard` dan melayani tanya-jawab operator.
Spesifikasi lengkap di `experiments/advisory/DESIGN.md`.

**Pembagian repo:** kode serving di `audiax_backend` (`internal/advisory/`),
artefak training di sini. Repo ini melatih, repo backend menyajikan. `ai/` dan
`service/` tidak tersentuh.

### Fase 0 — pemilihan base model (SELESAI)

Diukur pada 4 thread, sesuai batas `cpus: '4.0'` di `docker-compose.yml`:

| Model | p50 | p95 | JSON | tolak angka | tolak luar cakupan |
|---|---|---|---|---|---|
| gemma-3-270m-it-qat | 2,54 s | 4,35 s | 100%* | tidak | tidak |
| gemma-3-1b-it-qat | 6,59 s | 7,96 s | 100%* | ya | ya |

\* dengan grammar GBNF; tanpanya 270M menghasilkan 0/10 JSON valid.

**Keputusan: 1B ditolak.** Alasannya kecepatan, dan fine-tuning tidak bisa
memperbaiki kecepatan. Gemma 4 E2B gugur lebih awal — 4,3 GB tidak muat di
batas `memory: 4G`.

**Temuan yang mengubah desain:** grammar GBNF menjamin validitas JSON secara
struktural. Validitas JSON karenanya keluar dari daftar yang perlu dipelajari
fine-tuning; yang tersisa hanya perilaku isi. Beban belajar menyempit jauh, dan
itulah yang membuat model 270M layak dicoba sama sekali.

**Kegagalan isi model dasar yang jadi target korpus:**
- Membantah keputusan sistem: *"Mesin masih aman dipakai"* pada WARNING yang
  gerbangnya menyuruh mematikan mesin
- Mengarang definisi: *"crest factor adalah indikator kualitas kalibrasi, makin
  tinggi makin baik"* — salah, dan terbalik
- Tidak mengeskalasi: operator melapor bau gosong, dijawab checklist biasa

### Fase 2 — korpus (SELESAI)

657 giliran / 228 konteks. Split **per konteks**: train 518 / val 66 / test 73.
94% jawaban unik. Distribusi mendekati target `CORPUS_SPEC.md`.

Tiga niat penolakan (`bahaya`, `di_luar_cakupan`, `pancingan_angka`) awalnya
hanya 10–15 kasus karena sampling proporsional justru menekan yang paling
penting. Mode `--focus` menaikkannya ke 71–88.

### Fase 3 — LoRA (BERJALAN)

r=16 pada 7 proyeksi, 3,8M dari 272M parameter (1,40%). CPU, 195 langkah,
~3,5 jam. Checkpoint tiap 20 langkah dengan resume otomatis.

### Fase 5 — evaluasi (GERBANG LULUS)

Diukur pada 30 kasus test set di checkpoint 20 dari 195 — belum genap
sepertiga epoch pertama:

| Metrik | Dasar | + LoRA |
|---|---|---|
| JSON terbaca | 0/30 (0%) | 30/30 (100%) |
| Gerbang keselamatan | 18/30 (60%) | 28/30 (93%) |
| Tanpa angka asing | 30/30 (100%) | 30/30 (100%) |
| Tanpa frasa diagnosis | 29/30 (97%) | 30/30 (100%) |
| Eskalasi saat bahaya | 0/3 (0%) | 1/3 (33%) |

**Go.** Angka final menyusul setelah 195 langkah. Detail dan catatan
kejujurannya di `experiments/advisory/results.md`.

### Deployment (TERBUKTI)

`merged safetensors -> GGUF f16 526 MB -> q4_k_m 249 MB -> llama-server ->
jawaban benar`. Cukup kecil untuk di-bake ke image.

Kuirk Gemma 3 yang harus ditangani: tokenizer 262145 token vs embedding 262144
baris. Menaikkan `vocab_size` di config membuat konversi lolos tapi llama.cpp
menolak — metadata jadi berbohong soal bentuk tensor. Yang benar
`resize_token_embeddings()`, sudah otomatis di `merge_ckpt.py`.

Dua cacat yang belum selesai: flag `eskalasi` belum ikut benar walau teksnya
sudah benar (dampak produksi kecil — eskalasi diputuskan deterministik di Go),
dan ada kebocoran frasa prompt ke jawaban.

---

## 4. Masalah yang diketahui

| # | Masalah | Dampak |
|---|---|---|
| 1 | Checkpoint belum di-host | **Demo gagal di mesin juri.** Prioritas tertinggi |
| 2 | Belum ada `experiments/results.md` | Tidak ada angka AUC yang bisa dirujuk proposal |
| 3 | Kontradiksi compliance antar-repo | Lihat §5 |
| 4 | `experiments/` berisi 6 notebook | `CLAUDE.md` menyebut "satu notebook referensi" |
| 5 | Kuota Gemini tier gratis habis harian | Korpus berhenti di 657; cukup untuk gerbang |

---

## 5. Kontradiksi compliance — perlu keputusan tim

Terkonfirmasi dari kode, bukan dugaan:

| Bukti di `audiax_backend` | Klaim di `CLAUDE.md` / proposal |
|---|---|
| `db/migrations/0001..0004` — 4 tabel | *"Dilarang infrastruktur database"* |
| `middleware/auth.go` + sesi Redis | *"MVP tanpa sistem otentikasi"* |
| `entity.Inspection` menyimpan tiap hasil | *"Dilarang pencatatan data otomatis"* |
| `entity.Baseline` tersimpan di server | *"Baseline disimpan di klien"* |

Pilihannya dua: revisi klaim proposal agar jujur, atau revisi backend agar
sesuai klaim. Membiarkan keduanya berbeda adalah yang terburuk — juri yang
membuka repo akan menemukannya, dan itu menjatuhkan seluruh bab compliance.

---

## 6. Catatan arsitektur

- **Train/serve skew ditemukan dan ditutup.** Penghapusan blok `RIWAYAT:`
  menyisakan dua baris kosong di sisi Python sementara `prompt.go` menggabungkannya
  jadi satu. Golden file ada di
  `audiax_backend/internal/advisory/testdata/golden_prompt_no_history.txt`;
  **test Go yang membandingkannya belum ditulis** — masih perlu dikerjakan Track A.
- `prompt_template.txt` adalah sumber kebenaran tunggal, di-`go:embed` oleh Go
  dan dibaca langsung oleh generator korpus.
- Rule engine yang memutuskan; LLM hanya menjelaskan. Guard menolak angka dan
  klaim di luar fakta terukur, dan gagal-guard berarti fallback statis, bukan error.

---

## 7. File yang diubah di sesi ini

```
.gitignore                  hapus *.md dan /docs; tambah artefak training
.dockerignore               kecualikan duplikat .pt di root
Dockerfile                  verifikasi aset saat build
docker-compose.yml          hapus mount yang menutupi isi image
docker-compose.dev.yml      BARU — mount dev sebagai override eksplisit
scripts/download_assets.py  benar-benar mengunduh + exit non-zero
README.md                   ditulis lengkap
CLAUDE.md                   di-rename dari "CLAUDE (1).md"
docs/PROGRESS.md            BARU — file ini
experiments/advisory/       BARU — DESIGN, CORPUS_SPEC, KICKOFF_BACKEND,
                            gen_corpus, train_lora, eval_advisory, bench_phase0,
                            corpus/ (657 giliran)
```

---

## 8. Langkah berikutnya

1. **Unggah checkpoint ke GitHub Release** ← penghalang demo, hanya tim yang bisa
2. Tunggu training selesai → jalankan `eval_advisory.py` → **gerbang go/no-go**
3. Kalau go: konversi GGUF, tambah service Ollama ke compose, implementasi
   `provider.go` sisi lokal
4. Kalau no-go: `static.go` yang dikirim; korpus tetap jadi bukti proses
5. Track A: tulis test Go yang membandingkan render prompt dengan golden file
6. Putuskan kontradiksi compliance (§5)
7. Jalankan notebook untuk menghasilkan `experiments/results.md`
