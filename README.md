# audiax_model

Komponen AI AUDIAX — pemantauan kondisi mesin berbasis suara untuk UMKM pangan
kering. Repo ini **hanya** berisi model dan lapisan HTTP tipis di atasnya;
frontend dan backend ada di repo terpisah.

Alurnya: operator merekam ±2 menit suara mesin dalam kondisi sehat (kalibrasi,
sekali per unit), lalu rekaman harian ±10 detik dibandingkan terhadap baseline
itu. Keluarannya satu `HealthCard`: NORMAL / WARNING / CRITICAL /
KALIBRASI_KURANG.

---

## ⚠️ Langkah wajib sebelum menjalankan apa pun

```bash
python scripts/download_assets.py
```

Skrip ini menyiapkan dua aset yang **tidak ada di dalam repo**:

| Aset | Kenapa tidak di repo |
|---|---|
| `ai/vendor/beats/*.py` | Source BEATs milik Microsoft, di-vendor dari upstream |
| `ai/weights/beats_finetuned.pt` | 361 MB — di atas batas file GitHub (100 MB) |

Skrip **exit non-zero** kalau ada yang belum lengkap, jadi aman dipakai di CI.

Checkpoint diunduh dari URL yang kamu setel sendiri:

```bash
export AUDIAX_CHECKPOINT_URL=https://github.com/<org>/<repo>/releases/download/<tag>/beats_finetuned.pt
python scripts/download_assets.py
```

Atau salin manual file `.pt` ke `ai/weights/beats_finetuned.pt`.

---

## Menjalankan sebagai service (jalur demo)

```bash
python scripts/download_assets.py     # sekali, butuh internet
docker compose up --build             # setelah ini, tidak butuh internet lagi
```

Checkpoint dan vendor source **ikut ter-bake ke dalam image**, bukan di-mount
dari host. Setelah image jadi, container berjalan sepenuhnya offline — itu
persyaratan demo, karena koneksi di lokasi penjurian tidak bisa diandalkan.

Kalau aset belum lengkap, **build gagal dengan pesan yang menjelaskan langkahnya**,
bukan menghasilkan image yang lolos build tapi rusak saat dijalankan.

Cek kesiapan:

```bash
curl http://localhost:8000/healthz     # {"status":"ok"} setelah warm-up
```

Endpoint:

| Method | Path | Isi |
|---|---|---|
| POST | `/v1/calibrate` | multipart: `audio`, `machine_label` → `MachineBaseline` JSON |
| POST | `/v1/inspect` | multipart: `audio`, `baseline_json` → `HealthCard` JSON |
| GET | `/healthz` | 200 kalau model sudah termuat, 503 kalau belum |

Service ini **stateless**: ia mengembalikan baseline ke pemanggil dan menunggu
baseline itu dikirim balik pada tiap inspeksi. Tidak ada database di repo ini.

### Mengganti checkpoint tanpa rebuild (khusus pengembangan)

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

Override ini mengembalikan volume mount. **Jangan dipakai untuk demo** — mount
menutupi isi image, jadi kalau folder host kosong, service gagal warm-up.

---

## Menjalankan tanpa server

```bash
pip install -r requirements.txt
python -m ai.demo
```

Butuh Python 3.11. Python 3.13+ belum tentu punya wheel PyTorch yang cocok.

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
ai/            logika inti, nol dependensi web        -- DI-DEPLOY
service/       lapisan HTTP tipis (FastAPI)           -- DI-DEPLOY
tests/         boundary, pipeline, kontrak HTTP
experiments/   notebook riset + advisory layer        -- TIDAK DI-DEPLOY
scripts/       penyiap aset
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
