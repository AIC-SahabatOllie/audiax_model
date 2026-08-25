# Hasil eksperimen — inti anomaly detection (Blok A–E)

Semua angka di file ini diambil dari **output yang tersimpan di dalam notebook**,
bukan diketik ulang dari ingatan. Tiap tabel menyebut notebook dan bagian
asalnya, jadi bisa dibuka dan dicocokkan. Angka apa pun di proposal yang tidak
ada di sini **tidak boleh dipakai**.

> Hasil lapisan advisory (Teknisi Saku / Gemma-LoRA) ada di file terpisah:
> `experiments/advisory/results.md`. Jangan campur keduanya — beda komponen,
> beda dataset, beda metrik.

---

## 0. Notebook mana yang berlaku

`experiments/audiax_pipeline.ipynb` adalah notebook referensi tunggal
(sebelumnya bernama `audiax_pipeline_FRD_optimized.ipynb`). Notebook lain
dipindah ke `experiments/archive/` — bukan dibuang, tapi **tidak boleh** jadi
sumber angka. Silsilah dan alasannya ada di `experiments/archive/README.md`.

Konfigurasi run yang menghasilkan angka di bawah (Bagian 5, cell 16):

```
dataset_mode=local   dataset_root=../Dataset   RUN_TAG=local_mac_opt
holdout_machine_id="id_04"          <- Keputusan Desain #2 terpenuhi
Jalur data: audio_channel=None (rata-rata semua kanal), normalize_mode=peak, privacy_notch=True
Anti-overfit: pooling=mean_std, LP-FT=True (3 ep), L2-SP=0.001, EMA=True, seleksi=synth_auc
Objective adapter: MAC (machine_id), head=arcface scale=16.0 margin=0.1
                   -- label normal/abnormal TIDAK dipakai untuk training adapter
```

Baris terakhir itu berarti **Keputusan Desain #1 sudah terpenuhi** — status
"BELUM SELESAI" di `CLAUDE.md` sudah usang. Adapter dilatih dengan Machine
Attribute Classification atas `machine_id`; label anomali hanya dipakai untuk
menghitung AUC saat evaluasi.

---

## 1. Protokol evaluasi — baca ini dulu sebelum tabel mana pun

Notebook memisahkan dua populasi, dan **hanya satu yang sah dikutip** sebagai
klaim generalisasi:

| | Isi | Boleh dikutip sebagai generalisasi? |
|---|---|---|
| **P-A** | Unit yang adapternya ikut dilatih (id_00, id_02, id_06) | ❌ Tidak |
| **P-B** | Unit yang tidak pernah dilihat adapter (id_04) | ✅ Ya |

Bahkan di P-A, `evaluate_machine()` menerima `exclude_paths` dan B2 memanggilnya
dengan `ADAPTER_TRAIN_PATHS`, sehingga unit seen tidak dinilai pada klip yang
persis dipakai melatihnya.

**Rata-rata seluruh unit adalah jebakan.** Notebook mencetak peringatannya
sendiri: *"Rata-rata per stage — SELURUH unit (campuran seen + unseen, JANGAN
dikutip untuk B2)"*. Rata-rata itu naik karena tiga dari empat unit pernah
dilihat adapter. Mengutipnya sebagai bukti generalisasi adalah kesalahan yang
akan langsung terlihat oleh juri yang membuka notebook.

---

## 2. Hasil utama — ablasi B0 → B1 → B2

Sumber: `audiax_pipeline.ipynb` Bagian 30, agregasi `mean`.

### Per unit

| Unit | Dilihat adapter | B0 zero-shot cosine | B1 zero-shot fusi | B2 fine-tuned fusi |
|---|---|---|---|---|
| id_00 | ya | 0,5648 | 0,5659 | 0,6464 |
| id_02 | ya | 0,5889 | 0,5978 | 0,6878 |
| **id_04** | **tidak** | **0,5303** | **0,5333** | **0,5170** |
| id_06 | ya | 0,7174 | 0,7245 | 0,8647 |

### Rata-rata

| Populasi | B0 | B1 | B2 |
|---|---|---|---|
| Seluruh unit — **jangan dikutip untuk B2** | 0,6004 | 0,6054 | 0,6790 |
| **P-B (id_04, satu-satunya angka sah)** | **0,5303** | **0,5333** | **0,5170** |

### Yang sebenarnya dikatakan tabel ini

**Fine-tuning tidak memperbaiki unit yang belum pernah dilihat — malah sedikit
memperburuk.** 0,5333 → 0,5170 pada id_04. Kenaikan besar di id_00/id_02/id_06
(sampai +0,14) adalah kenaikan pada unit yang adapternya ikut dilatih, dan itu
bukan generalisasi.

Ini temuan negatif, dan ditulis apa adanya karena satu-satunya hal yang lebih
buruk daripada temuan negatif adalah temuan negatif yang ditemukan juri lebih
dulu. Yang tetap berdiri:

- **Fusi 4-backend konsisten ≥ cosine saja** di keempat unit (B1 ≥ B0 selalu,
  meski tipis: +0,001 s/d +0,009). Fusi minimum tidak pernah merugikan.
- **Pipeline zero-shot (B1) pada id_04 = 0,5333** sudah berada di rentang yang
  sama dengan baseline resmi Hitachi untuk unit itu (0,5715), tanpa training
  sama sekali.

### Perbandingan ke baseline resmi

Baseline DAE resmi `MIMII-hitachi/mimii_baseline`, kategori fan @ −6 dB:

| Unit | Baseline resmi | B1 zero-shot | B2 fine-tuned |
|---|---|---|---|
| id_00 | 0,5757 | 0,5659 | 0,6464 |
| id_02 | 0,6401 | 0,5978 | 0,6878 |
| **id_04** | **0,5715** | 0,5333 | 0,5170 |
| rata-rata 3 unit | **≈0,596** | 0,5657 | 0,6504 |

⚠️ **Caveat traceability yang belum tertutup.** Run ini memakai
`DATASET_MODE="local"` dengan root `../Dataset`, bukan mode `"mimii"` yang
mengunduh `-6_dB_fan` resmi. Struktur unitnya identik (id_00/02/04/06, 150 klip
normal per unit di test) dan Bagian 26b notebook sendiri membandingkan id_04 ke
angka 0,5715, tapi **tingkat SNR salinan lokal itu belum diverifikasi ulang**.
Sampai ada yang menjalankan mode `"mimii"` dan mencocokkan, perbandingan di
tabel ini harus disebut *indikatif*, bukan setara apple-to-apple.

---

## 3. Metrik operasional pada ambang produksi

Bukan cuma AUC — ini perilaku pada ambang yang benar-benar dipakai
`ai/decision.py`. Sumber: Bagian 30, "Ringkasan metrik operasional B2".

| Unit | Dilihat | AUC | pAUC | FPR warn | Recall warn | FPR crit | Recall crit |
|---|---|---|---|---|---|---|---|
| id_00 | ya | 0,6464 | 0,5698 | 0,102 | 0,268 | 0,011 | 0,057 |
| id_02 | ya | 0,6878 | 0,6119 | 0,087 | 0,329 | 0,011 | 0,102 |
| **id_04** | **tidak** | **0,5170** | **0,4991** | 0,096 | **0,073** | 0,011 | **0,013** |
| id_06 | ya | 0,8647 | 0,8324 | 0,031 | 0,680 | 0,000 | 0,543 |

**Ini angka yang paling penting dan paling tidak nyaman.** Pada unit baru,
recall WARNING hanya **7,3%** dan recall CRITICAL **1,3%**. Artinya: dengan
ambang produksi saat ini, pada unit yang belum pernah dilihat, sistem
**melewatkan lebih dari 90% kondisi abnormal**.

False positive rate-nya terkendali (9,6% warn / 1,1% crit) — jadi kesalahannya
bukan "rewel", tapi "diam". Untuk alat triase, diam saat ada masalah lebih
berbahaya daripada rewel.

Konsekuensi untuk klaim produk: posisikan sebagai **alat bantu triase yang
konservatif** — kalau berbunyi, layak dicek; kalau diam, **bukan** jaminan
sehat. `HealthCard` sudah wajib membawa disclaimer, dan angka ini adalah
alasan kuantitatifnya.

---

## 4. Tiga objective adapter yang pernah dicoba

| Objective | Notebook | id_04 B0 | id_04 B2 | Delta |
|---|---|---|---|---|
| Focal Loss biner (normal/abnormal) | `archive/..._AOT_New.ipynb` §30 | 0,5221 | **0,5718** | +0,050 |
| MAC cross-entropy linear | `archive/..._AOT_New_MAC.ipynb` §30 | 0,5221 | 0,5509 | +0,029 |
| MAC + ArcFace + LP-FT/L2-SP/EMA/WiSE-FT | `audiax_pipeline.ipynb` §30 | 0,5303 | 0,5170 | **−0,013** |

**Notebook yang paling canggih menghasilkan angka held-out paling buruk.**

Dua hal harus disebut bersamaan dengan tabel ini:

1. **Perbandingan ini belum sah.** Notebook `_optimized` memakai protokol
   evaluasi berbeda (`exclude_paths`, `per_unit_test_frac`) *dan* jalur data
   berbeda (`privacy_notch=True`, `normalize_mode=peak`) — terlihat dari B0-nya
   sendiri yang sudah beda (0,5303 vs 0,5221). Jadi selisih B2 mencampur efek
   objective dengan efek protokol dan preprocessing. Notebook menamai pekerjaan
   ini **T5 — "adu adil"** dan mencatatnya sebagai **belum dikerjakan**.
2. **Focal Loss biner melanggar Keputusan Desain #1.** Angka 0,5718 adalah yang
   terbaik di kolom itu, tapi diperoleh dengan melatih adapter memakai label
   anomali — persis yang dilarang premis *training-free ASD*. **Tidak boleh
   dikutip sebagai hasil sistem.** Dicatat di sini hanya supaya keputusan
   berpindah ke MAC tidak terlihat seperti keputusan yang menyembunyikan biaya.

Plafon yang jujur untuk MAC di dataset ini: `machine_id` adalah satu-satunya
atribut yang tersedia, jadi ruang label MAC hanya **3 kelas** (unit seen).
Setup MAC di DCASE memakai banyak atribut per klip. Dengan 3 kelas, tugas proxy
terlalu mudah dan sinyal yang dipelajari terbatas — ini penjelasan struktural,
bukan pembelaan.

---

## 5. Ablasi denoise HPSS

Sumber: Bagian 29. Unit uji id_04, agg `mean`, klip training adapter dikecualikan.

| | AUC | pAUC |
|---|---|---|
| `enable_denoise=False` (default sekarang) | 0,4907 | 0,4931 |
| `enable_denoise=True` | **0,5282** | 0,4933 |
| Delta | **+0,0375** | +0,0002 |

Ini **ablasi nyata pertama** untuk HPSS pada data non-sintetis — Keputusan
Desain #4 mensyaratkan tepat ini sebelum default boleh diubah. Arahnya positif
dan cukup besar pada AUC.

**Tetap: jangan ubah default sekarang.** Tiga alasan:

1. pAUC praktis tidak bergerak (+0,0002). Perbaikan terjadi di wilayah FPR
   tinggi yang tidak dipakai ambang produksi. Yang menentukan pengalaman
   pengguna adalah wilayah FPR rendah, dan di situ denoise tidak membantu.
2. Satu unit, satu run, tanpa ulangan seed. +0,0375 belum bisa dibedakan dari
   derau run-to-run.
3. HPSS menambah biaya komputasi di jalur inferensi yang berjalan di ponsel.

Kalau ada waktu: ulangi pada keempat unit dan ≥3 seed. Kalau delta bertahan
**dan** pAUC ikut naik, baru ubah default.

---

## 6. Sapuan alpha WiSE-FT

Sumber: Bagian 16b. Interpolasi bobot pretrained ↔ fine-tuned untuk melawan
*feature distortion*.

| alpha | AUC seleksi (anomali sintetis, unit seen) | Held-out id_04 (laporan saja) |
|---|---|---|
| 0,00 | 0,7969 | 0,4481 |
| 0,10 | 0,7962 | 0,4630 |
| 0,20 | 0,7919 | 0,4830 |
| 0,30 | 0,7975 | 0,4995 |
| 0,50 | 0,8150 | **0,5089** |
| 0,70 | 0,8250 | 0,4970 |
| **1,00 ← terpilih** | **0,8381** | 0,4592 |

Kriteria seleksi adalah AUC anomali sintetis — **bebas label anomali asli**,
sehingga pemilihan alpha tidak membocorkan test set. Itu metodologi yang benar
dan harus dipertahankan.

Tapi tabel ini juga menunjukkan **kriteria seleksinya tidak berkorelasi dengan
yang kita pedulikan**: kriteria memilih alpha=1,00 (held-out 0,4592) padahal
alpha=0,50 memberi held-out terbaik 0,5089. Di ujung atas, kedua kolom justru
bergerak berlawanan arah.

Kolom held-out **tidak boleh** dipakai untuk memilih alpha — itu akan menjadi
kebocoran test set. Yang benar adalah mencari kriteria seleksi bebas-label yang
lebih baik. Anomali sintetis yang dipakai sekarang jelas tidak mewakili anomali
nyata pada unit baru.

---

## 7. Diagnostik kemiripan antar-unit

Sumber: Bagian 26b, dihitung dari embedding **zero-shot** sehingga tidak bisa
dipengaruhi pilihan split.

Jarak cosine antar-centroid unit:

```
        id_00   id_02   id_04   id_06
id_00  0.0000  0.0067  0.0067  0.0192
id_02  0.0067  0.0000  0.0051  0.0163
id_04  0.0067  0.0051  0.0000  0.0147
id_06  0.0192  0.0163  0.0147  0.0000
```

Sebaran **dalam**-unit: id_00 0,0366 · id_02 0,0302 · id_04 0,0385 · id_06 0,0304.

**Jarak antar-unit 3–7× lebih kecil daripada sebaran di dalam satu unit.**
Keempat unit praktis menempati wilayah embedding yang sama; variasi antar-klip
dalam satu unit jauh melampaui variasi antar-unit.

Ini menjelaskan dua hal sekaligus:

- **Kenapa MAC hanya memberi sedikit.** Tugas proxy "tebak unit mana" terlalu
  mudah dipecahkan lewat isyarat dangkal ketika kelasnya cuma 3 dan centroidnya
  berdempetan — gradiennya tidak memaksa backbone mempelajari struktur yang
  berguna untuk membedakan sehat vs menyimpang.
- **Kenapa kalibrasi per-instance adalah keputusan arsitektur yang benar.**
  Kalau perbedaan antar-unit saja sekecil ini, ambang global mustahil bekerja.
  Membandingkan tiap mesin ke baseline dirinya sendiri bukan penyederhanaan —
  itu satu-satunya cara yang masuk akal secara geometris. **Ini pembenaran
  kuantitatif terkuat untuk desain inti AUDIAX, dan layak masuk proposal.**

---

## 8. Yang belum dikerjakan

| # | Item | Kenapa penting |
|---|---|---|
| 1 | **T5 — adu adil binary vs MAC** | Tabel §4 belum sah tanpa ini |
| 2 | **Bagian 29b belum pernah dijalankan** (cell tanpa output) | 8 kombinasi `audio_channel` × `normalize_mode` × `privacy_notch` belum terukur |
| 3 | **Verifikasi `Dataset/` = MIMII −6 dB fan** | Menutup caveat §2 |
| 4 | **Denoise: ulang di 4 unit × ≥3 seed** | Syarat sebelum ubah Keputusan Desain #4 |
| 5 | **Kriteria seleksi bebas-label yang lebih baik** | §6 menunjukkan yang sekarang tidak berkorelasi |
| 6 | **Recall pada unit baru** | 7,3% warn / 1,3% crit — batasan produk paling nyata |

Item 1–3 murni menjalankan ulang notebook (butuh dataset + GPU). Item 6 adalah
masalah riset terbuka, bukan bug.

---

## 9. Cara menghasilkan ulang

```bash
# Butuh: GPU, dataset multi-unit di ../Dataset/id_XX/{normal,abnormal}/*.wav,
#        checkpoint BEATs iter3+ pretrained di models/
jupyter lab experiments/audiax_pipeline.ipynb
# Jalankan berurutan. Bagian 5 = konfigurasi; DATASET_MODE dan
# holdout_machine_id ada di situ. Bagian 30 mencetak tabel §2 file ini.
```

Notebook **tidak** dipanggil `service/` maupun `docker compose` — sesuai
Batasan Rulebook #4. Ia riset dan referensi, bukan bagian jalur inferensi.

Artefak CSV yang dihasilkan (`ablation_results_local_mac_opt.csv`,
`unit_similarity_local_mac_opt.csv`) tidak ikut ter-commit karena butuh dataset
untuk dibuat ulang; angkanya sudah disalin ke file ini.
