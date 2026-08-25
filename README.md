# audiax_model

Komponen AI AUDIAX — pemantauan kondisi mesin berbasis suara untuk UMKM pangan
kering. Repo ini **hanya** berisi model dan lapisan HTTP tipis di atasnya;
frontend dan backend ada di repo terpisah.

Alurnya: operator merekam ±2 menit suara mesin dalam kondisi sehat (kalibrasi,
sekali per unit), lalu rekaman harian ±10 detik dibandingkan terhadap baseline
itu. Keluarannya satu `HealthCard`: NORMAL / WARNING / CRITICAL /
KALIBRASI_KURANG.

---

## 🚀 Panduan Setup (untuk panitia — menjalankan secara lokal)

Ada dua jalur untuk mencoba model ini. **Jalur A (Docker)** direkomendasikan —
sekali build, lalu jalan sepenuhnya offline, cocok untuk lokasi penjurian
dengan koneksi tidak stabil. **Jalur B (tanpa server)** lebih cepat kalau
Docker tidak tersedia dan cukup mencoba lewat command line.

### Prasyarat

| Jalur | Butuh |
|---|---|
| A — Docker | Docker Desktop (Compose v2), ~5 GB ruang disk kosong, koneksi internet **hanya saat setup** |
| B — tanpa server | Python 3.11 (Python 3.13+ belum tentu punya wheel PyTorch yang cocok), ~3 GB ruang disk kosong |

Kedua jalur butuh langkah 1 di bawah lebih dulu (butuh internet, sekali saja).

### 1. Unduh aset yang tidak ada di repo (wajib, sekali, butuh internet)

```bash
python scripts/download_assets.py
```

Skrip ini menyiapkan dua aset yang **tidak ikut ter-clone**:

| Aset | Kenapa tidak di repo |
|---|---|
| `ai/vendor/beats/*.py` | Source BEATs milik Microsoft, di-vendor dari upstream |
| `ai/weights/beats_finetuned.pt` | 344,8 MB — di atas batas file GitHub (100 MB) |

Checkpoint diunduh otomatis dari GitHub Release repo ini — **tidak perlu
menyalin token atau menyetel apa pun**. Sumbernya
[`v0.1.0-weights`](https://github.com/AIC-SahabatOllie/audiax_model/releases/tag/v0.1.0-weights)
(sha256 diverifikasi otomatis oleh skrip). Skrip **exit non-zero** kalau ada
yang gagal/tidak lengkap — kalau perintah di atas selesai dengan pesan
`Semua aset siap.`, lanjut ke langkah berikutnya; kalau tidak, baca pesan error
yang dicetak (biasanya cukup jalankan ulang, unduhan bisa dilanjutkan dari titik
putus).

Kalau perlu sumber checkpoint lain (mirror internal, hasil training sendiri):

```bash
export AUDIAX_CHECKPOINT_URL=https://.../beats_finetuned.pt
python scripts/download_assets.py
```

atau salin manual file `.pt` ke `ai/weights/beats_finetuned.pt` (nama file harus
persis).

### 2A. Jalur A — jalankan sebagai service HTTP (Docker, direkomendasikan)

```bash
docker compose up --build     # dari sini seterusnya, tidak butuh internet lagi
```

Checkpoint dan vendor source **ikut ter-bake ke dalam image**, bukan di-mount
dari host — image yang sudah jadi berjalan sepenuhnya offline. Kalau aset di
langkah 1 belum lengkap, **build gagal dengan pesan yang menjelaskan langkahnya**,
bukan menghasilkan image yang lolos build tapi rusak saat dijalankan.

Tunggu sampai container sehat (warm-up model, biasanya beberapa detik–puluhan
detik tergantung mesin), lalu cek:

```bash
curl http://localhost:8000/healthz     # {"status":"ok"} setelah warm-up selesai
```

Coba alur kalibrasi + inspeksi memakai sampel yang sudah ada di
`sample_dataset/` (5 klip `normal/`, 5 klip `abnormal/`):

```bash
curl -X POST http://localhost:8000/v1/calibrate \
  -F "audio=@sample_dataset/normal/00000000.wav" \
  -F "machine_label=Blower Oven Demo" \
  -o baseline.json

curl -X POST http://localhost:8000/v1/inspect \
  -F "audio=@sample_dataset/abnormal/00000000.wav" \
  -F "baseline_json=<baseline.json" \
  | python -m json.tool
```

Endpoint lengkap:

| Method | Path | Isi |
|---|---|---|
| POST | `/v1/calibrate` | multipart: `audio`, `machine_label` → `MachineBaseline` JSON |
| POST | `/v1/inspect` | multipart: `audio`, `baseline_json` → `HealthCard` JSON |
| GET | `/healthz` | 200 kalau model sudah termuat, 503 kalau belum |

Service ini **stateless**: ia mengembalikan baseline ke pemanggil dan menunggu
baseline itu dikirim balik pada tiap inspeksi (lihat `baseline.json` di atas) —
tidak ada database di repo ini.

Matikan container setelah selesai:

```bash
docker compose down
```

#### Mengganti checkpoint tanpa rebuild (khusus pengembangan)

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

Override ini mengembalikan volume mount. **Jangan dipakai untuk demo** — mount
menutupi isi image, jadi kalau folder host kosong, service gagal warm-up.

### 2B. Jalur B — jalankan tanpa server (CLI langsung)

```bash
pip install -r requirements.txt
python -m ai.demo --calibrate sample_dataset/normal/00000000.wav \
  --label "Blower Oven Demo" --baseline-out baseline.json
python -m ai.demo --inspect sample_dataset/abnormal/00000000.wav \
  --baseline-in baseline.json
```

Hasilnya dicetak langsung ke terminal (status, z-score, health score, indikator
dominan, disclaimer). `python -m ai.demo --info` mencetak info model tanpa
perlu file audio — berguna untuk memverifikasi checkpoint termuat dengan benar.

### Troubleshooting singkat

| Gejala | Penyebab paling mungkin |
|---|---|
| `download_assets.py` exit non-zero | Koneksi terputus di tengah unduhan — jalankan ulang, ia melanjutkan dari titik putus, bukan mengulang dari nol |
| `docker compose up --build` gagal di step verifikasi aset | Langkah 1 belum dijalankan atau gagal sebagian — cek `ai/weights/` dan `ai/vendor/beats/` |
| `/healthz` balas 503 terus | Model masih warm-up (tunggu `start_period` healthcheck) — kalau tidak kunjung sehat, cek `docker compose logs audiax-ai` |
| `pip install` gagal cari wheel `torch` | Python 3.13+ — pakai Python 3.11 |

## Test

```bash
pytest tests/
```

- `tests/test_boundary.py` — menegakkan `ai/` bebas dependensi web
- `tests/test_pipeline.py` — logika Blok A–E dengan encoder tiruan (tidak butuh checkpoint)
- `tests/test_service_routes.py` — kontrak HTTP tanpa model asli

---

## Struktur

```
ai/              logika inti, nol dependensi web        -- DI-DEPLOY
service/         lapisan HTTP tipis (FastAPI)           -- DI-DEPLOY
tests/           boundary, pipeline, kontrak HTTP
experiments/     notebook riset + advisory layer        -- TIDAK DI-DEPLOY
scripts/         penyiap aset
sample_dataset/  klip contoh (normal/abnormal) untuk coba cepat, lihat Panduan Setup
```

Pendekatan teknis: BEATs (pretrained AudioSet, **beku**) → adapter ringan yang
di-fine-tune → kalibrasi per unit mesin → scoring 4-backend (cosine, Mahalanobis,
kNN, PCA) dengan fusi minimum → ambang statistik per-instance.

## Batas klaim

- **Bukan** *Acoustic Emission* dalam pengertian NDT — itu butuh sensor
  piezoelektrik 100 kHz–1 MHz. Yang dianalisis adalah tanda vibro-akustik pada
  rentang audible yang tertangkap mikrofon ponsel.
- **Bukan** alat diagnosis yang mengikat — alat bantu **triase**. Setiap
  `HealthCard` menyertakan disclaimer ini.
- **Bukan** classifier jenis kerusakan. `dominant_indicator` hanya menyebut fitur
  DSP mana yang paling bergeser — petunjuk arah, bukan label kerusakan.

## Lisensi

MIT — lihat `LICENSE`. Catatan: source dan checkpoint BEATs tunduk pada lisensi
Microsoft masing-masing, dan sebagian varian MIMII berlisensi CC BY-NC-SA.
Periksa keduanya sebelum penggunaan komersial.
