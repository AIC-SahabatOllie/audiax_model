# Arsip notebook — silsilah, bukan sampah

Notebook di folder ini **tidak dibuang** tapi juga **bukan sumber angka**.
Satu-satunya notebook yang berlaku adalah `experiments/audiax_pipeline.ipynb`.
Angka yang boleh dikutip ada di `experiments/results.md`.

Alasan folder ini ada: `CLAUDE.md` menetapkan "satu notebook referensi", tapi
repo sempat berisi enam notebook dengan nama berbeda dan tidak ada satu pun
bernama `audiax_pipeline.ipynb`. Siapa pun yang membaca `CLAUDE.md` lalu
membuka `experiments/` akan menemukan ketidakcocokan itu langsung. Menghapus
lima notebook akan menghilangkan jejak eksperimen yang justru memperkuat
metodologi, jadi keduanya dipisah: satu berlaku, lima terarsip.

---

## Dua garis keturunan yang berbeda

### Garis BEATs — inilah arsitektur AUDIAX

Sesuai `CLAUDE.md`: BEATs beku → adapter → kalibrasi per-instance → 4 backend
→ fusi minimum.

| Urutan | Notebook | Objective adapter | Kenapa digantikan |
|---|---|---|---|
| 1 | `audiax_pipeline_nic.ipynb` | Focal Loss biner | Draf awal, `DATASET_MODE="custom"` (satu unit), sebagian besar cell tanpa output. Tidak bisa mengukur generalisasi antar-unit sama sekali |
| 2 | `audiax_pipeline_AOT_New.ipynb` | Focal Loss biner | Melanggar Keputusan Desain #1 — adapter dilatih memakai label anomali, bertentangan dengan premis *training-free ASD* |
| 3 | `audiax_pipeline_AOT_New_MAC.ipynb` | MAC cross-entropy linear | Keputusan Desain #1 terpenuhi, tapi head linear mengoptimalkan separabilitas linear sementara keempat backend scoring membaca geometri cosine/jarak — objective dan pemakaian tidak selaras |
| **4** | **`../audiax_pipeline.ipynb`** (dulu `_FRD_optimized`) | **MAC + ArcFace** | **Berlaku.** Protokol evaluasi P-A/P-B, `exclude_paths`, pAUC, metrik ambang produksi, seleksi model bebas-label |

Angka id_04 (held-out) dari ketiga varian terlatih dibandingkan di
`results.md` §4 — termasuk fakta tidak nyaman bahwa varian tercanggih justru
menghasilkan angka held-out terburuk, dan kenapa perbandingan itu belum sah.

### Garis STgram-MFN — arsitektur alternatif, bukan iterasi

| Notebook | Isi |
|---|---|
| `audiax_pipeline_FRD_refactor.ipynb` | STgram-MFN + Sub-Center ArcFace, ruang label MAC |
| `audiax_pipeline_FRD_refactor2.ipynb` | STgram-MFN satu-kelas, tiga kepala bebas label |

**Keduanya tidak memakai BEATs sama sekali.** Front-end ganda Sgram+Tgram dengan
backbone MobileFaceNet, dilatih dari nol. Ini eksplorasi arsitektur pembanding,
bukan tahap berikutnya dari garis BEATs.

Angkanya tidak sebanding dengan `results.md` — beda encoder, beda split, beda
protokol evaluasi. Jangan menaruhnya di tabel yang sama. Kalau ada yang ingin
mengangkat jalur ini, yang dibutuhkan lebih dulu adalah menjalankan keduanya di
harness evaluasi P-A/P-B yang sama.

---

## Aturan

- ❌ Jangan mengutip angka dari folder ini ke proposal atau `results.md`
- ❌ Jangan menyinkronkan Blok A–E ke notebook di folder ini (kewajiban sinkron
  di `CLAUDE.md` hanya berlaku untuk `experiments/audiax_pipeline.ipynb`)
- ✅ Boleh dibuka untuk menelusuri kenapa sebuah keputusan diambil
