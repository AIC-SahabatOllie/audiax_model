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
| Lapisan advisory (Teknisi Saku) | ✅ Gerbang lulus, model terlatih, artefak GGUF 249 MB |
| **Angka AUC dari notebook** | ✅ **`experiments/results.md` sudah ada** (lihat §4) |
| **Konsolidasi notebook** | ✅ **Satu notebook berlaku, lima terarsip** (lihat §4) |
| **Test golden prompt (Go)** | ✅ **Ditulis, dan menemukan bug skew nyata** (lihat §5) |

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

### ⚠️ Sisa penghalang demo — hanya tim yang bisa

`beats_finetuned.pt` (345 MB) belum di-host di mana pun. Orang yang meng-clone
repo tidak punya cara mendapatkannya. **Unggah ke GitHub Release**, lalu
`AUDIAX_CHECKPOINT_URL` di README berfungsi. Sampai itu dikerjakan, jalur "juri
clone lalu jalankan" masih putus. Semua sisi kode sudah siap menerimanya.

---

## 3. Lapisan advisory — Teknisi Saku

Fitur pendamping: menjelaskan `HealthCard` dan melayani tanya-jawab operator.
Spesifikasi lengkap di `experiments/advisory/DESIGN.md`, angka di
`experiments/advisory/results.md`.

**Pembagian repo:** kode serving di `audiax_backend` (`internal/advisory/`),
artefak training di sini. Repo ini melatih, repo backend menyajikan. `ai/` dan
`service/` tidak tersentuh.

### Ringkas hasil

| Fase | Hasil |
|---|---|
| 0 — pemilihan base | 270M dipilih; 1B ditolak (p95 7,96 s); Gemma 4 E2B gugur (4,3 GB > batas 4G) |
| 2 — korpus | 657 giliran / 228 konteks, 94% unik, split per konteks 518/66/73 |
| 3 — LoRA | 3,8M dari 272M parameter (1,40%), dihentikan di 1 dari 3 epoch |
| 5 — evaluasi | Test set penuh 73 kasus, **gerbang lulus** |
| Deployment | GGUF q4_k_m 249 MB, terbukti jalan di llama-server |

Gerbang keselamatan **45,2% → 97,3%**; eskalasi saat bahaya **0/9 → 6/9**.
Model dasar tidak pernah sekalipun mengeskalasi saat operator melaporkan bau
gosong, asap, atau panas berlebih.

Empat batasan kejujuran ditulis eksplisit di `experiments/advisory/results.md`
dan **wajib ikut dikutip** bersama angkanya.

### Artefak

`experiments/advisory/dist/audiax-advisor-q4km.gguf` — 249 MB, dibangun dari
checkpoint-60 yang persis dievaluasi. File `.gguf` tidak masuk git; resep
lengkap di `dist/README.md`.

Kuirk Gemma 3 yang harus ditangani: tokenizer 262145 token vs embedding 262144
baris. Menaikkan `vocab_size` di config membuat konversi lolos tapi llama.cpp
menolak — metadata jadi berbohong soal bentuk tensor. Yang benar
`resize_token_embeddings()`, sudah otomatis di `merge_ckpt.py`.

Dua cacat yang belum selesai: flag `eskalasi` belum ikut benar walau teksnya
sudah benar (dampak produksi kecil — eskalasi diputuskan deterministik di Go),
dan ada kebocoran frasa prompt ke jawaban.

---

## 4. Notebook & angka AUC — SELESAI di sesi ini

### Masalah yang ditemukan

`CLAUDE.md` menyebut `experiments/audiax_pipeline.ipynb` sebagai satu-satunya
notebook referensi. **File itu tidak pernah ada.** Yang ada enam notebook dengan
nama berbeda. Akibatnya `experiments/results.md` tidak pernah bisa dibuat —
tidak jelas notebook mana yang otoritatif — dan siapa pun yang membaca
`CLAUDE.md` lalu membuka `experiments/` akan menemukan ketidakcocokan langsung.

### Yang dikerjakan

```
experiments/audiax_pipeline.ipynb      <- dulu _FRD_optimized, satu-satunya yang berlaku
experiments/archive/                   <- lima pendahulu + README silsilah
experiments/results.md                 <- BARU, angka dari output tersimpan di notebook
```

Notebook `_FRD_optimized` dipilih karena ia satu-satunya yang punya protokol
evaluasi P-A/P-B (`exclude_paths`), pAUC, metrik ambang produksi, dan objective
MAC. Dua notebook `_FRD_refactor*` memakai arsitektur **berbeda sama sekali**
(STgram-MFN, tanpa BEATs) — itu eksplorasi pembanding, bukan iterasi.

### Keputusan Desain #1 ternyata SUDAH selesai

`CLAUDE.md` menandai objective MAC sebagai "⚠️ BELUM SELESAI". Notebook
`_FRD_optimized` cell 16 mencetak sendiri:

> `Objective adapter: MAC (machine_id), head=arcface scale=16.0 margin=0.1 --`
> `label normal/abnormal TIDAK dipakai untuk training adapter.`

`CLAUDE.md` sudah diperbarui: #1 ✅ selesai, #4 diberi angka ablasi nyata.

### Temuan angka yang harus diketahui tim

**Fine-tuning tidak memperbaiki unit yang belum pernah dilihat.** Pada held-out
id_04: B1 zero-shot 0,5333 → B2 fine-tuned **0,5170**. Kenaikan besar yang
terlihat di rata-rata (0,6054 → 0,6790) berasal dari tiga unit yang adapternya
ikut dilatih — notebook sendiri mencetak *"JANGAN dikutip untuk B2"* di sebelah
angka itu.

**Recall pada unit baru sangat rendah**: WARNING 7,3%, CRITICAL 1,3%. FPR
terkendali (9,6% / 1,1%). Jadi kesalahan sistem bukan "rewel" tapi "diam" —
dan untuk alat triase, diam lebih berbahaya. Ini harus tercermin di klaim
produk: kalau berbunyi layak dicek, kalau diam **bukan** jaminan sehat.

**Yang tetap kuat dan layak masuk proposal**: jarak antar-unit di ruang embedding
3–7× lebih kecil daripada sebaran di dalam satu unit (Bagian 26b). Itu bukti
kuantitatif bahwa ambang global mustahil bekerja, dan karenanya **kalibrasi
per-instance adalah keputusan arsitektur yang benar**, bukan penyederhanaan.

Detail dan enam item yang belum dikerjakan ada di `experiments/results.md` §8.

---

## 5. Test golden prompt — SELESAI, dan menemukan bug nyata

Golden file `audiax_backend/internal/advisory/testdata/golden_prompt_no_history.txt`
sudah ada sejak sesi lalu tapi **tidak ada test yang memakainya**. Sekarang ada:
`golden_prompt_test.go`, lima test, membandingkan **byte-per-byte**.

Menulisnya menyingkap tiga masalah:

1. **Golden file lama ditulis tangan dan salah di tiga tempat** — `urgensi`
   memakai spasi (`rencanakan dalam 48 jam`) padahal korpus memakai garis bawah
   (`rencanakan_dalam_48_jam`); checklist terpotong 2 dari 4 butir;
   `eskalasi_bila` diringkas. Test terhadap golden yang salah lebih buruk
   daripada tidak ada test — ia akan memaksa Go menyesuaikan diri ke format yang
   model tidak pernah lihat.
2. **Train/serve skew nyata di produksi.** `gen_corpus.py` memanusiakan tiga
   enum sebelum merender (`belt` → `sabuk-puli`, `>6bln` → `>6 bulan`,
   `3-5th` → `3-5 tahun`), tapi `prompt.go` menulis kodenya apa adanya. Pemanggil
   di `usecase/advisory.go` meneruskan `request.Context.DriveType` mentah — jadi
   model yang dilatih pada `penggerak: sabuk-puli` akan menerima
   `penggerak: belt` di produksi. Diperbaiki: peta label dipindah ke `prompt.go`,
   plus test yang memindai `decision_table.json` dan gagal kalau ada kode baru
   tanpa label.
3. **Golden tidak boleh diketik tangan lagi.** `experiments/advisory/gen_golden.py`
   merendernya lewat `gen_corpus.py::render_prompt` — fungsi yang persis sama
   dengan yang menghasilkan tiap contoh training, jadi mustahil keduanya beda.

Diverifikasi bahwa test-nya benar-benar bisa gagal: humanisasi dimatikan
sementara, test langsung gagal dengan pesan yang menunjuk baris 6
(`"penggerak: sabuk-puli"` vs `"penggerak: belt"`), lalu dikembalikan.

Seluruh suite backend hijau: `go test ./...` → 5 paket ok.

---

## 6. Masalah yang diketahui

| # | Masalah | Dampak |
|---|---|---|
| 1 | Checkpoint belum di-host | **Demo gagal di mesin juri.** Prioritas tertinggi, hanya tim yang bisa |
| 2 | Kontradiksi compliance antar-repo | Lihat §7 — perlu keputusan tim |
| 3 | Recall unit baru 7,3% / 1,3% | Batasan produk nyata, harus tercermin di klaim |
| 4 | T5 "adu adil" binary vs MAC belum dijalankan | Tabel `results.md` §4 belum sah |
| 5 | Bagian 29b notebook belum pernah dijalankan | 8 kombinasi jalur data belum terukur |
| 6 | `Dataset/` belum diverifikasi = MIMII −6 dB fan | Perbandingan ke anchor resmi masih indikatif |
| 7 | Kuota Gemini tier gratis habis harian | Korpus berhenti di 657; cukup untuk gerbang |
| 8 | `internal/config/config.go` di repo backend | Perubahan dari sesi lain, **tidak** saya commit — `godotenv.Load` sekarang menelan semua error, termasuk `.env` yang rusak. Perlu ditinjau pemiliknya |

---

## 7. Kontradiksi compliance — perlu keputusan tim

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

## 8. Catatan arsitektur

- `prompt_template.txt` adalah sumber kebenaran tunggal, di-`go:embed` oleh Go
  dan dibaca langsung oleh `gen_corpus.py`. Kontrak render (8 aturan) di
  `audiax_backend/internal/advisory/PROMPT_CONTRACT.md`, ditegakkan golden test.
- Rule engine yang memutuskan; LLM hanya menjelaskan. Guard menolak angka dan
  klaim di luar fakta terukur, dan gagal-guard berarti fallback statis, bukan error.
- `ai/` tetap nol dependensi web (`tests/test_boundary.py`). Advisory tidak
  menyentuh `ai/` maupun `service/` sama sekali.

---

## 9. File yang diubah di sesi ini

### Repo ini (`audiax_model`)

```
experiments/audiax_pipeline.ipynb   di-rename dari audiax_pipeline_FRD_optimized.ipynb
experiments/archive/                BARU — 5 notebook pendahulu + README silsilah
experiments/results.md              BARU — angka AUC, traceable ke output notebook
experiments/advisory/gen_golden.py  BARU — regenerasi golden file, jangan diketik tangan
CLAUDE.md                           Keputusan Desain #1 -> selesai; #4 diberi angka ablasi;
                                    mode dataset "custom" -> "local"; catatan folder archive
docs/PROGRESS.md                    file ini
```

### Repo `audiax_backend`

```
internal/advisory/prompt.go              driveLabel/recencyLabel/ageLabel + humanise()
internal/advisory/golden_prompt_test.go  BARU — 5 test, byte-per-byte
internal/advisory/testdata/              BARU di git — golden diregenerasi, yang lama salah
internal/advisory/prompt_template.txt    BARU di git — sebelumnya untracked
internal/advisory/PROMPT_CONTRACT.md     BARU di git + aturan 8 (humanisasi) + bagian golden
```

Tidak disentuh: `internal/config/config.go`, `.air.toml` — bukan pekerjaan sesi ini.

---

## 10. Langkah berikutnya

1. **Unggah checkpoint ke GitHub Release** ← penghalang demo, hanya tim yang bisa
2. **Putuskan kontradiksi compliance** (§7) — ini bab penilaian, bukan detail teknis
3. Sesuaikan klaim produk dengan recall unit baru (§4) sebelum ditulis di proposal
4. Kalau ada GPU + waktu: jalankan T5 dan Bagian 29b, lalu perbarui `results.md`
5. Opsional: lanjutkan training LoRA dari checkpoint-60 (resume otomatis sudah ada).
   Kalau dilanjutkan, tabel hasil dan artefak `dist/` **harus** diregenerasi
   bersamaan supaya angka yang dipublikasikan tetap cocok dengan model yang dikirim
6. **Cabut dua Gemini API key** yang pernah ditempel di chat, setelah lomba selesai
