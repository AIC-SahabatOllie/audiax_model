# CLAUDE.md — Panduan Proyek AUDIAX (Repo AI/Model)

## 📁 Baca Dulu Sebelum Mulai

Setiap sesi baru, baca file-file ini **sebelum** menyentuh kode apapun:

### Dokumentasi di `/docs/`
- `docs/PROGRESS.md` — kondisi dan progres proyek terkini (snapshot kondisi saat ini, **bukan** log kumulatif — baca ini dulu untuk tahu apa yang sudah beres dan apa yang masih pending) dan jangan lupa update file ini setiap kali ada perubahan.
- `docs/model_implementation.md` — arsitektur AI, struktur modul, kontrak fungsi, spesifikasi tiap Blok A–E, alur kalibrasi/inspeksi

### Dokumen lain yang relevan
- `README.md` (root repo) — cara setup, menjalankan tanpa server (`python -m ai.demo`), menjalankan sebagai service HTTP, menjalankan test suite
- `experiments/audiax_pipeline.ipynb` — **satu-satunya** notebook referensi (pelatihan + evaluasi + walkthrough e2e lengkap), lihat §"Deployment vs Referensi" di bawah
- Proposal lengkap tim (di luar repo ini) — Bab 1–4 berisi latar belakang, tujuan, dan metodologi yang **menentukan** batasan teknis di bawah

---

## 🎯 Tentang Proyek: AUDIAX

**Masalah.** UMKM pangan kering (kerupuk, keripik, ikan asin) bergantung pada oven pengering bertenaga blower LPG. Kegagalan blower (bantalan aus, impeler kotor akibat debu tepung, ketidakseimbangan putaran akibat getaran) tidak terdeteksi dini karena usaha mikro tidak punya departemen maintenance, teknisi bersertifikat, atau anggaran condition monitoring — praktik yang berlaku saat ini sepenuhnya reaktif dan bertumpu pada pendengaran operator senior (pengetahuan tacit: tidak terdokumentasi, tidak terukur, hilang saat operator berganti).

**Solusi.** AUDIAX adalah sistem pemantauan kondisi mesin berbasis suara yang berjalan di smartphone yang **sudah** dimiliki operator (penetrasi internet 81,72%, 84,31% via smartphone — APJII 2026) — nol investasi perangkat keras tambahan, nol keahlian teknis khusus. Alurnya: operator merekam ±2 menit suara mesin dalam kondisi sehat (kalibrasi, sekali per unit mesin), lalu sistem membandingkan rekaman harian ±10 detik terhadap baseline itu dan mengeluarkan status NORMAL / WARNING / CRITICAL berdasarkan seberapa jauh tanda akustik bergeser dari kondisi sehatnya sendiri.

**Yang sistem ini BUKAN** (batasan klaim yang wajib dijaga di kode maupun komunikasi):
- **Bukan** deteksi *Acoustic Emission* dalam pengertian NDT klasik — itu memerlukan sensor piezoelektrik ultrasonik 100kHz–1MHz; mikrofon smartphone konsumer hanya menangkap hingga ±20–24kHz. Yang dianalisis adalah tanda **vibro-akustik pada rentang audible**.
- **Bukan** alat diagnosis yang mengikat — alat bantu **triase**. Setiap keluaran (`HealthCard`) wajib menyertakan disclaimer ini.
- **Bukan** classifier jenis kerusakan. MIMII (dataset publik yang dipakai untuk validasi) tidak menyediakan taksonomi kerusakan berlabel — yang ada hanya atribusi fitur DSP sebagai *petunjuk arah* (`ai/decision.py`).

**Pendekatan teknis inti.** BEATs (audio embedding pretrained AudioSet, **beku**) → adapter ringan yang di-fine-tune → kalibrasi instance-specific (memory bank per unit mesin, dibangun dari rekaman kalibrasi milik unit itu sendiri) → multi-backend anomaly scoring klasik (cosine, Mahalanobis, kNN, PCA) → fusi minimum → ambang statistik per-instance. Rezimnya seharusnya **training-free / first-shot anomaly detection** — lihat status belum-final di §"Keputusan Desain" #1.

Repo ini (`audiax-ai`) **hanya** berisi komponen AI. Frontend dan backend ada di repo terpisah — tiga deployment independen, dihubungkan lewat HTTP (lihat `service/main.py`).

---

## 📦 Deployment vs Referensi — Jangan Tertukar

Repo ini berisi **dua kategori artefak yang sangat berbeda perannya**:

| | Isi | Deploy ke Docker/HuggingFace? |
|---|---|---|
| `ai/*.py` + `service/*.py` | Kode produksi murni Python — preprocessing, encoder, adapter, kalibrasi, scoring, keputusan, lapisan HTTP | **Ya** |
| `experiments/audiax_pipeline.ipynb` | Fine-tuning adapter, evaluasi AUC, ablasi B0→B1→B2, DAN walkthrough alur inferensi lengkap langkah-demi-langkah — semuanya dalam **satu notebook** | **Tidak** — murni riset & referensi |

Notebook itu sengaja "melakukan semuanya" (training penuh sampai demo inferensi
lengkap) **karena** tidak dideploy — bebas berisi hal yang dilarang di kode
produksi (instalasi paket, plot, cell yang bergantung urutan eksekusi manual).
Kode produksi (`ai/`) sengaja jauh lebih ringkas karena itu yang harus lolos
batasan rulebook (statis, sinkron, tanpa auto-tuning — lihat bawah).

Notebook mendukung **dua mode dataset** (`DATASET_MODE` di Bagian 4):
`"mimii"` (mengunduh `-6_dB_fan` publik, bisa dibandingkan langsung ke baseline
resmi Hitachi) dan `"local"` (salinan multi-unit di `<root>/id_XX/{normal,abnormal}/*.wav`).
Angka di `experiments/results.md` dihasilkan mode `"local"` — perbandingan ke
anchor resmi karenanya masih **indikatif**, lihat caveat di `results.md` §2.

Lima notebook pendahulu ada di `experiments/archive/` — silsilahnya di
`experiments/archive/README.md`. Angkanya **tidak boleh** dikutip.

---

## 🔒 Batasan Rulebook Kompetisi — WAJIB Dipatuhi

Repo ini disubmit untuk penilaian teknis dengan batasan eksplisit dari dokumen "Teknis & Kriteria" panitia. Kode yang melanggar ini dinilai tidak memenuhi ruang lingkup MVP yang diizinkan:

1. **Model AI wajib fokus pada core inference dengan parameter yang bersifat statis saat demonstrasi berjalan.** Seluruh bobot (`BEATsEncoder`, `Adapter`) **harus beku** saat inferensi (`torch.inference_mode()`, `requires_grad = False`). Kalibrasi (`ai/calibration.py`) hanya menghitung statistik deskriptif (mean/std dari self-score leave-one-out) — **bukan** gradient descent, **bukan** fine-tuning saat runtime.
2. **Dilarang menyertakan sistem pembaruan otomatis (auto-tuning).** Ambang (`z_warning`, `z_critical`) diturunkan **sekali** saat kalibrasi dan dibekukan; tidak pernah menyesuaikan diri sendiri secara berkelanjutan.
3. **Dilarang pipeline pencatatan data otomatis (automated data logging) atau infrastruktur database terdistribusi.** Konsekuensinya, `service/` didesain **sepenuhnya stateless** — `MachineBaseline` disimpan di **klien**, bukan di server mana pun (lihat README §"Nol state di server"). Jangan menambahkan database apa pun ke repo ini untuk alasan apa pun tanpa mendiskusikan ulang batasan ini.
4. **Dilarang skrip pengujian massal (bulk testing scripts) pada repositori tahap penyisihan.** `experiments/audiax_pipeline.ipynb` sengaja **di luar** `docker compose` dan **tidak** dipanggil oleh `service/` — jangan pernah menautkannya ke Dockerfile atau ke alur inferensi produksi.
5. **Backend/API wajib hanya memproses interaksi sinkron.** `service/main.py` — satu permintaan, satu balasan, tanpa background job, tanpa worker, tanpa message queue.

---

## 🧭 Keputusan Desain yang Mengikat (Jangan Diubah Tanpa Diskusi)

Butir-butir berikut adalah hasil analisis/koreksi eksplisit di sesi-sesi sebelumnya dan **menentukan validitas metodologis proposal**. Kalau menemukan kode yang bertentangan dengan salah satu ini, itu kemungkinan besar **regresi**, bukan pembaruan yang disengaja — cek `docs/PROGRESS.md` dulu sebelum mengubah.

1. **✅ SELESAI — Objective adapter adalah Machine Attribute Classification (MAC), BUKAN binary classifier normal/abnormal.** `experiments/audiax_pipeline.ipynb` melatih adapter atas label `machine_id` dengan head ArcFace (`scale=16.0`, `margin=0.1`); klip abnormal sengaja tidak dipakai untuk training adapter sama sekali. Label anomali **hanya** masuk saat menghitung AUC di evaluasi. Ini memenuhi premis *training-free ASD* (lihat [17] Fang dkk. 2025, dan Bab 1.6 proposal). Varian Focal Loss biner yang lama diarsipkan di `experiments/archive/` dan **tidak boleh** dikutip sebagai hasil sistem walau angka held-out-nya kebetulan lebih tinggi — lihat `experiments/results.md` §4. Skala dan margin ArcFace **tetap** (bukan AdaCos) supaya tidak ada celah argumen soal auto-tuning.
2. **`holdout_machine_id = "id_04"`, bukan `id_06`** (mode dataset `"mimii"`). `id_06` adalah unit **termudah** di baseline resmi Hitachi (AUC 0.982 @ 0dB) — memakainya sebagai held-out membuat angka generalisasi optimistis dan mudah dipatahkan juri. `id_04` adalah unit **tersulit** (0.5715 @ −6dB).
3. **Anchor baseline AUC ≈ 0.60**, bukan 0.658. Angka yang benar adalah rata-rata baseline resmi `MIMII-hitachi/mimii_baseline` (DAE) untuk kategori **fan @ −6dB** (id_00=0.5757, id_02=0.6401, id_04=0.5715). Anchor ini **hanya berlaku untuk mode dataset `"mimii"`** — jangan dibandingkan langsung ke hasil mode `"custom"`.
4. **`enable_denoise` default `False`** di `ai/config.py`. Ablasi nyata Bagian 29 sudah ada (`experiments/results.md` §5): pada id_04, AUC 0,4907 → 0,5282 (**+0,0375**) tapi pAUC hanya 0,4931 → 0,4933 (**+0,0002**). Perbaikannya terjadi di wilayah FPR tinggi yang tidak dipakai ambang produksi, diukur pada satu unit tanpa ulangan seed, dan HPSS menambah biaya komputasi di ponsel. **Default tetap `False`.** Syarat untuk mengubahnya: ulangi di 4 unit × ≥3 seed, dan pAUC harus ikut naik — bukan cuma AUC.
5. **4 backend scoring (cosine, Mahalanobis+shrinkage, kNN density-normalized, PCA residual) difusi dengan MINIMUM** — bukan rata-rata, bukan voting, bukan pemilihan backend "terbaik" otomatis. Minimum bersifat konservatif (semua backend harus sepakat baru dianggap anomali), sesuai temuan [15] Zhou & Wang (2026).
6. **3 repo, 3 deployment terpisah** (FE / BE / AI — repo ini). Konsekuensinya: `ai/` = pure Python library (nol dependensi web, ditegakkan `tests/test_boundary.py`), `service/` = lapisan HTTP tipis **di atas** `ai/`. Jangan menggabungkan lagi jadi satu proses/image kecuali ada keputusan eksplisit untuk kembali ke arsitektur monorepo.
7. **Bukan 3 model penuh terpisah** (gate suara-mesin / normalizer / predictor). Layer "gate suara-mesin vs bukan" bersifat **opsional**, memakai embedding BEATs yang sama (linear probe ringan), dan **baru** diaktifkan kalau data audio non-mesin sudah tersedia.
8. **Satu notebook referensi, bukan dua.** `experiments/audiax_pipeline.ipynb` menggabungkan pelatihan + evaluasi + walkthrough e2e. Jangan memecahnya lagi jadi notebook terpisah tanpa alasan kuat — penggabungan ini keputusan eksplisit supaya seluruh alur (dari data mentah sampai `HealthCard`) bisa dibaca sebagai satu cerita utuh.

---

## 🚫 Aturan Wajib — Jangan Dilanggar

- ❌ **JANGAN `git commit`**
- ❌ **JANGAN `git push`**
- ❌ **JANGAN operasi git apapun yang mengubah history** (merge, rebase, reset, checkout yang menimpa perubahan, dll.)
- Git hanya boleh dipakai untuk **membaca**: `git status`, `git diff`, `git log`.
- ❌ **JANGAN mengimpor framework web (FastAPI, Flask, Starlette, dll.) di dalam `ai/`.** Ditegakkan otomatis oleh `tests/test_boundary.py` — kalau test ini gagal, perbaiki import yang menyebabkannya, **jangan** menghapus atau melemahkan test-nya.
- ❌ **JANGAN commit checkpoint (`*.pt`) atau vendor source BEATs (`ai/vendor/beats/*.py`) ke git.** Sudah masuk `.gitignore`; kalau `git status` menunjukkan file itu ter-track, itu bug yang harus dilaporkan, bukan di-force-add.
- ❌ **JANGAN tambah dependensi baru tanpa memperbarui file `requirements.txt` yang tepat.** Ada empat: `ai/requirements.txt` (kode inti), `service/requirements.txt` (lapisan HTTP), `experiments/requirements.txt` (notebook riset), `requirements.txt` di root (konsolidasi, untuk instalasi dev sekali jalan). Perbarui **semua** file yang relevan, bukan cuma satu.
- ❌ **JANGAN menambah komponen (model/backend/layer) baru tanpa test di `tests/`.** Minimal: satu test logika (pakai encoder tiruan dari `tests/conftest.py`, tidak perlu checkpoint BEATs asli).
- ❌ **JANGAN mengubah logika Blok A–E di `ai/*.py` tanpa menyinkronkan salinannya di `experiments/audiax_pipeline.ipynb` Bagian 6, 20–22** (dan sebaliknya). Notebook standalone (tidak `import ai` langsung, lihat §"Deployment vs Referensi"), jadi kedua sisi bisa diam-diam menyimpang kalau tidak disiplin. Catat setiap sinkronisasi di `docs/PROGRESS.md`.

---

## 🏗️ Arsitektur — Ringkasan Cepat

> Detail lengkap dan spesifikasi tiap modul ada di `docs/model_implementation.md`.

### Stack
Python 3.11 · PyTorch + torchaudio (+ `soundfile` untuk baca audio di notebook — lebih tahan terhadap masalah backend `torchcodec` yang pernah muncul di sebagian sesi Colab) · Encoder: **BEATs iter3+ (AS2M)**, Microsoft, beku · FastAPI untuk lapisan HTTP · pytest untuk testing · Docker multi-stage untuk deployment.

### 4 Layer, 1 yang Belajar
```
Preprocessing (DSP klasik)  ->  Encoder BEATs (beku)  ->  Adapter (SATU-SATUNYA bobot terlatih)  ->  Scoring 4-backend + fusi minimum  ->  Keputusan (ambang instance-specific)
```

### Struktur Folder
```
ai/            <- logika inti, NOL dependensi web (ditegakkan test_boundary.py) -- DI-DEPLOY
service/       <- lapisan HTTP tipis (FastAPI) -- SATU-SATUNYA yang boleh impor web framework -- DI-DEPLOY
tests/         <- test_boundary.py (arsitektur), test_pipeline.py (logika, encoder tiruan),
                   test_service_routes.py (kontrak HTTP, tanpa model asli)
experiments/   <- audiax_pipeline.ipynb: training + evaluasi + walkthrough e2e -- TIDAK DI-DEPLOY
docs/          <- PROGRESS.md, model_implementation.md
```

### Prinsip Kode
- Type hints di semua fungsi/class publik; docstring menjelaskan **kenapa**, bukan cuma **apa**.
- Setiap perubahan yang menyentuh preprocessing/scoring/threshold **wajib** disertai test yang memakai encoder tiruan (`tests/conftest.py`).
- Kalau ragu antara menambah fitur baru vs menjaga scope tetap kecil: **pilih scope kecil**. Rubrik kompetisi eksplisit menilai "apakah ruang lingkup MVP sudah tepat, tidak overbuilt atau underbuilt".
- Angka AUC/hasil eksperimen apa pun yang dirujuk di proposal **harus** bisa ditelusuri balik ke `experiments/results.md` atau `experiments/ablation_results.csv` — jangan menuliskan angka yang tidak pernah benar-benar dihasilkan notebook.

---

## ✅ Kewajiban Setelah Setiap Sesi

Setelah setiap sesi kerja selesai, **wajib tulis ulang** `docs/PROGRESS.md` dengan kondisi terkini. Ini **bukan** log kumulatif — isinya adalah **snapshot kondisi repo saat ini** (checklist per modul, rencana kelanjutan & handoff, masalah/bug yang diketahui, catatan arsitektur, daftar file yang diubah di sesi ini). Format mengikuti template yang sudah ada di `docs/PROGRESS.md` — pertahankan strukturnya, cukup perbarui isinya.
