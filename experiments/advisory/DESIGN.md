# DESIGN — Teknisi Saku (Advisory Layer)

**Tanggal:** 2026-08-24
**Status:** disetujui, siap dieksekusi
**Anggaran waktu:** 24 jam, dua jalur paralel
**Ringkas:** lapisan pendamping berbasis Gemma hasil fine-tune LoRA, berjalan lokal
(offline) di dalam Docker, yang menjelaskan HealthCard dan melayani tanya-jawab
lanjutan operator — tanpa pernah memutuskan apa pun dan tanpa pernah membuat angka.

---

## 1. Keputusan yang sudah dikunci

| # | Keputusan | Alasan |
|---|---|---|
| 1 | Kode serving hidup di `audiax_backend`, sebagai modul terkurung `internal/advisory/` | Repo AI wajib nol dependensi web (`tests/test_boundary.py`). Rubrik menghukum overbuild, jadi bukan repo/service ke-4. |
| 2 | Repo AI (`ai/`, `service/`) **tidak diubah sama sekali** | Batas arsitektur yang sudah ada tetap tegak |
| 3 | Artefak training hidup di `audiax_model/experiments/advisory/` | `experiments/` sudah dikecualikan dari Docker. Repo model melatih, repo backend menyajikan. |
| 4 | Model: Gemma hasil LoRA, dijalankan via container Ollama, GGUF **di-bake ke image** | Klausul rulebook "model wajib di-fine-tune"; sekaligus menjaga klaim `docker compose up` tanpa internet |
| 5 | Ukuran base model **belum dikunci** — ditentukan Fase 0 | Tidak ada benchmark publik untuk kualitas Bahasa Indonesia maupun kecepatan CPU di ukuran ini |
| 6 | LLM tidak pernah memutuskan. Rule engine yang memutuskan. | Keputusan harus deterministik dan bisa di-unit-test |
| 7 | Guard output tetap aktif **walaupun** model sudah di-fine-tune | Fine-tune menurunkan pelanggaran; guard membuatnya nol |
| 8 | Gagal guard ⇒ jatuh ke `StaticProvider`, bukan error | Fitur tidak pernah mati; degradasi jujur lewat field `source` |

**Batas klaim (wajib dijaga, sejalan `CLAUDE.md`):** ini alat bantu triase, bukan
diagnosis. Advisory tidak pernah menyebut jenis kerusakan, tidak pernah
menghasilkan angka di luar fakta terukur, tidak pernah memprediksi sisa umur mesin.

---

## 2. Jadwal paralel (jam ke-, dari T+0)

Ketergantungan kritis: `decision_table.json` adalah **input** prompt teacher.
Karena itu Track A memproduksinya lebih dulu (Fase 1a), bukan di akhir Fase 1.

```
      Track A (Go / Backend)                Track B (Python / Colab)
T+0   1a decision_table.json  (2j)          F0 Spike 3 model       (2j)  <-- GO/NO-GO
T+2   ────────────── serah terima decision_table.json ──────────────
T+2   1b rules+guard+provider (5j)          F2 Korpus              (4j)
T+6                                         F3 LoRA 2 ukuran       (3j)
T+7   1c endpoint+wiring      (1j)
T+8   (selesai; bantu Track B)
T+9                                         F4 GGUF + Modelfile    (2j)
T+11  ───────── integrasi bersama: ollama.go + docker-compose (1j) ─────────
T+12  ───────── F5 harness evaluasi + tabel metrik (3j) ─────────
T+15  SELESAI. Buffer 9 jam.
```

**Jalur kritis = Track B (14 jam).** Track A selesai di T+8 dan setelah itu
menjadi cadangan tenaga. Buffer 9 jam sengaja disisakan untuk video, proposal,
dan hal yang meleset.

**Aturan main:** setiap fase punya *definition of done* di §7. Fase tidak boleh
ditinggalkan setengah jadi — lebih baik berhenti di batas fase daripada
menggantung di tengah dua fase.

---

## 3. Kontrak bersama (WAJIB disepakati sebelum T+0)

Dua orang membangun terpisah terhadap kontrak ini. Perubahan apa pun di sini
harus diumumkan ke dua-duanya.

### 3.1 `decision_table.json` — dihasilkan Track A, dikonsumsi Track A & B

```json
{
  "schema_version": "1.0",
  "cells": [
    {
      "key": "WARNING|crest_factor|belt|>6bln",
      "urgency": "rencanakan_dalam_48_jam",
      "safety_gate": "Matikan mesin dan tunggu impeler berhenti total sebelum memeriksa apa pun.",
      "checklist": [
        "Periksa ketegangan sabuk - tekan di tengah, lendutan wajar sekitar 1 cm per 100 cm jarak puli.",
        "Lihat permukaan sabuk: retak, mengkilap, atau serat terkelupas.",
        "Raba rumah bearing (mesin mati): terasa jauh lebih panas dari biasanya?",
        "Cek baut dudukan motor dan blower, kencangkan yang longgar."
      ],
      "escalate_if": "Kalau keempat hal di atas normal tapi besok statusnya tetap WARNING, panggil teknisi.",
      "recheck_hours": 48,
      "needs_technician": false
    }
  ]
}
```

Kunci sel: `status|dominant_indicator|drive_type|recency`.

**Wildcard `*` diizinkan di dimensi mana pun.** Ruang penuh 4 dimensi =
4 × 4 × 3 × 4 = 192 kombinasi, tapi sebagian besar tidak mengubah jawaban —
mis. saat `NORMAL`, tidak ada dimensi lain yang relevan. Wildcard menjaga tabel
tetap ~40 sel yang bisa di-review manusia satu per satu, bukan 192 yang tidak
akan pernah dibaca siapa pun.

Contoh:
```
"NORMAL|*|*|*"                    -> 1 sel menutup 48 kombinasi
"KALIBRASI_KURANG|*|*|*"          -> 1 sel
"WARNING|crest_factor|belt|*"     -> 1 sel menutup 4 kombinasi
"WARNING|crest_factor|belt|>6bln" -> lebih spesifik, menang atas baris di atas
```

**Aturan pencocokan:** pilih sel dengan **jumlah wildcard paling sedikit** yang
cocok. Bila seri (mustahil bila tabel benar), ambil yang muncul lebih dulu dan
catat sebagai bug. `rules_test.go` wajib memuat test yang memverifikasi:
(a) setiap dari 192 kombinasi menemukan tepat satu sel, (b) tidak ada sel yang
tidak pernah terpakai (dead cell).

Target: ~40 sel. Semua sel ditulis konservatif — bila ragu, arahkan ke teknisi.

### 3.2 Prompt yang dirakit Go dan dilihat model

Format ini identik saat training dan saat serving. **Train/serve skew di sini
akan menghancurkan hasil** — sama seriusnya dengan skew di pipeline akustik.

```
FAKTA (tidak boleh diubah atau ditambah):
  status: WARNING
  z: 3.4 (ambang: warning 3.0, kritis 6.0)
  indikator_dominan: crest_factor
  kualitas_kalibrasi: baik
  penggerak: sabuk-puli
  terakhir_dirawat: >6 bulan
  umur_mesin: 3-5 tahun
  jam_operasi_harian: >8
  ada_cadangan: tidak

KEPUTUSAN SISTEM (tidak boleh diubah):
  urgensi: rencanakan dalam 48 jam
  gerbang_keselamatan: Matikan mesin dan tunggu impeler berhenti total...
  langkah:
    1. Periksa ketegangan sabuk...
    2. Lihat permukaan sabuk...
    3. Raba rumah bearing...
    4. Cek baut dudukan...
  eskalasi_bila: Kalau keempat hal di atas normal tapi besok tetap WARNING...
  periksa_lagi_dalam_jam: 48

RIWAYAT:
  operator: crest factor itu apa?
  asisten: Itu ukuran seberapa tajam puncak suara dibanding suara rata-ratanya...

PERTANYAAN: sabuknya udah kenceng kok, tapi tadi ada bau gosong dikit
```

Riwayat dibatasi **8 giliran terakhir**. Field yang bernilai kosong dihilangkan
barisnya (bukan ditulis `null`).

### 3.3 Output model — JSON, wajib

```json
{
  "jawaban": "string, 2-4 kalimat, bahasa operator",
  "langkah_berikutnya": "string, satu kalimat perintah konkret",
  "perlu_teknisi": true,
  "eskalasi": true
}
```

`eskalasi: true` artinya model menilai informasi baru dari operator lebih
mendesak dari status awal. **Nilai ini hanya saran** — keputusan eskalasi yang
mengikat dibuat deterministik di Go (§3.5).

### 3.4 Kontrak HTTP (Backend ↔ Flutter)

**Fakta klinis TIDAK dikirim klien.** Server mengambilnya sendiri dari tabel
`inspections` berdasarkan `:inspectionId`, dengan pengecekan kepemilikan lewat
`machines.user_id`. Kalau klien boleh mengirim `status`, klien bisa memalsukan
`NORMAL` pada mesin yang CRITICAL — dan seluruh rantai keselamatan runtuh.
Klien hanya mengirim apa yang memang miliknya: riwayat percakapan, pertanyaan,
dan atribut mesin.

`calibration_quality` **tidak ada** di tabel `inspections` (lihat
`internal/entity/inspection.go`). Ambil dari baris `baselines` lewat
`inspections.baseline_id`. Bila join gagal, hilangkan barisnya dari prompt —
jangan tulis `null`.

```
POST /api/machines/:machineId/inspections/:inspectionId/advisory/messages
Authorization: Bearer <token>

Request:
{
  "history": [{"role": "user|assistant", "content": "..."}],
  "user_message": "sabuknya udah kenceng tapi ada bau gosong",
  "context": {
    "drive_type": "belt|direct-coupled|direct-drive",
    "recency": "<1bln|1-6bln|>6bln|tidak-tahu",
    "machine_age": "<1th|1-3th|3-5th|>5th",
    "hours_per_day": "<4|4-8|>8",
    "has_backup": false,
    "load_state": "kosong|bermuatan"
  }
}
```

Atribut di `context` belum ada kolomnya di tabel `machines`
(`internal/entity/machine.go` hanya punya label/location/description). Untuk
sekarang **dikirim klien** dan di-cache lokal oleh FE — sengaja, supaya fitur ini
tidak terblokir oleh perubahan CRUD mesin. Migrasi `0005_add_machine_attributes`
masuk daftar lanjutan pasca-lomba, dan saat itu `context` jadi opsional
(override) alih-alih wajib.

```

Response 200:
{
  "reply": "...",
  "next_step": "...",
  "needs_technician": true,
  "escalated": true,
  "source": "llm|fallback_static",
  "disclaimer": "Alat bantu triase, bukan diagnosis mengikat -- tetap perlu inspeksi teknisi."
}
```

`source` **wajib** ditampilkan FE (badge kecil). Degradasi harus terlihat, bukan disembunyikan.

### 3.5 Kata kunci bahaya — deterministik, tanpa LLM

Dicek di Go pada `user_message` sebelum LLM dipanggil. Cocok = paksa jalur
CRITICAL, `escalated=true`, `needs_technician=true`, dan gerbang keselamatan
dipasang apa pun status akustiknya.

```
bau gosong, bau terbakar, asap, berasap, percikan, api,
panas berlebih, sangat panas, getaran keras, bergetar hebat,
bunyi ledakan, macet, tersendat
```

### 3.6 Kontrak Ollama (Backend ↔ container model)

```
POST http://ollama:11434/api/chat
{
  "model": "audiax-advisor",
  "messages": [{"role": "user", "content": "<prompt §3.2>"}],
  "stream": false,
  "format": "json",
  "options": {"temperature": 0, "top_k": 1, "top_p": 1, "seed": 42, "num_predict": 220}
}
```

Timeout 5 detik. Lewat timeout ⇒ `StaticProvider`.

---

## 4. Track A — Backend Go

### 4.0 Konvensi repo yang WAJIB diikuti

Dibaca langsung dari kode, bukan diasumsikan:

| Hal | Aturan di repo ini |
|---|---|
| Framework | **Fiber v2** (bukan Gin). Module `audiax`, Go 1.25 |
| Konstanta | **Semua** konstanta hidup di `internal/constants/constants.go`. Package doc-nya eksplisit: *"If you are about to write `const` anywhere else, write it here instead."* Jadi kata kunci bahaya, path Ollama, dan timeout masuk ke sana — **bukan** file `danger_keywords.go` sendiri |
| Klien HTTP keluar | Polanya `internal/config/aiservice.go`. Klien Ollama mengikuti pola itu di `internal/config/llmservice.go`, bukan di dalam `internal/advisory/` |
| Error domain | `internal/apperr` (`ErrUnavailable`, `RejectedError`) |
| Test | `*_test.go` bersebelahan dengan kode, `testify`, fake di `fakes_test.go` |
| Layering | delivery → usecase → repository; usecase tidak tahu HTTP |

### 4.1 Struktur file

```
internal/advisory/            <- PAKET MURNI, nol dependensi
    decision_table.json       # go:embed
    rules.go                  # lookup (status, indicator, drive, recency) -> Cell
    prompt.go                 # perakit prompt §3.2
    guard.go                  # validator §4.2
    provider.go               # interface LLMProvider {Complete(ctx, prompt) (string, error)}
    static.go                 # fallback dari Cell
    rules_test.go  guard_test.go  prompt_test.go  boundary_test.go

internal/config/llmservice.go # implementasi Ollama, meniru aiservice.go
internal/constants/           # + kata kunci bahaya, OllamaChatPath, AdvisoryTimeout
internal/delivery/http/advisory_controller.go
internal/usecase/advisory.go  # + advisory_test.go
internal/model/advisory.go
internal/delivery/http/route/route.go   # + 1 rute
```

**Kenapa implementasi Ollama tidak ikut di `internal/advisory/`:** paket itu harus
tetap murni supaya `boundary_test.go` bisa menjaga batasnya (§4.3). `advisory`
mendefinisikan *interface*-nya; `config` yang menyediakan implementasinya dan
menyuntikkannya — persis pola `AIService` yang sudah ada.

Konstanta baru di `internal/constants/constants.go`:
```go
// Advisory layer.
const (
    OllamaChatPath   = "/api/chat"
    OllamaModelName  = "audiax-advisor"
    // Jauh lebih pendek dari DefaultAITimeout (240s): forward pass BEATs memang
    // lambat, tapi balasan 220 token dari model 1B yang belum selesai dalam 5
    // detik tidak akan selesai, dan operator sedang menunggu.
    DefaultAdvisoryTimeout = 5 * time.Second
    AdvisoryMaxHistoryTurns = 8
    AdvisoryNumPredict      = 220
)

// DangerKeywords memaksa jalur CRITICAL tanpa melibatkan LLM (§3.5).
var DangerKeywords = []string{...}
```


### 4.2 Guard — empat pemeriksaan, semua wajib lolos

| Guard | Aturan | Gagal ⇒ |
|---|---|---|
| Skema | Output parse sebagai JSON sesuai §3.3 | fallback |
| Angka | Setiap angka di `jawaban`+`langkah_berikutnya` harus ada di whitelist: nilai dari HealthCard, dari Cell, dan ordinal 1-9 | fallback |
| Diagnosis | Tidak boleh cocok daftar frasa terlarang: `bearing aus`, `bearing rusak`, `impeler pecah`, `motor terbakar`, `kerusakan pada`, `disebabkan oleh`, `sisa umur`, `akan rusak dalam` | fallback |
| Keselamatan | Bila `Cell.safety_gate` tidak kosong, output wajib memuat instruksi mematikan mesin sebelum instruksi memeriksa | fallback |

### 4.3 Batas modul (`boundary_test.go`)

`internal/advisory/` **dilarang** meng-import: GORM, `internal/entity`,
`internal/config`, `internal/repository`, `internal/usecase`, dan framework HTTP
(Fiber). Menerima struct polos, mengembalikan struct polos.

**Diizinkan:** pustaka standar dan `internal/constants` — paket itu sendiri tidak
meng-import apa pun dari modul ini (lihat package doc-nya), jadi tidak bisa
menyeret dependensi masuk.

Test memindai import dan gagal bila dilanggar. Ini pola yang sama dengan
`tests/test_boundary.py` di repo AI, dan simetri itu layak disebut di proposal:
batas arsitektur ditegakkan test di kedua repo, bukan lewat konvensi.

---

## 5. Track B — Training

### 5.1 Fase 0 — Spike (2 jam) — GERBANG GO/NO-GO

Kandidat: **Gemma 3 270M**, **Gemma 3 1B**, **Gemma 4 E2B** (Apache 2.0).

Ukur di CPU yang menyerupai laptop juri, memakai GGUF siap pakai (belum di-tune):
1. tok/s dan latensi p95 untuk balasan ~150 token
2. RAM terpakai
3. Kualitas Bahasa Indonesia pada 10 prompt uji (penilaian manusia, kasar)
4. **Uji konversi GGUF sekarang juga**, jangan tunggu Fase 4

| Hasil | Aksi |
|---|---|
| Ada kandidat p95 < 5 dtk DAN Indonesia layak | Kunci model itu, lanjut |
| Semua > 5 dtk tapi 270M layak bahasanya | Pakai 270M, potong `num_predict` |
| Semua gagal | **Batalkan jalur Gemma**, pindah ke dev-time only. Rugi 2 jam, bukan 14. |

### 5.2 Fase 2 — Korpus (4 jam)

600 konteks sampel bertingkat dari ruang penuh 9.216, × 3-4 giliran = ~2.000 giliran.

| Niat pertanyaan | Porsi |
|---|---|
| Tanya arti istilah | 15% |
| Boleh jalan atau tidak | 15% |
| Lapor observasi baru | 20% |
| Sudah kerjakan langkah, minta lanjutan | 15% |
| Kapan panggil teknisi / biaya | 10% |
| Di luar cakupan → tolak sopan | 8% |
| Pancingan halusinasi → wajib menolak beri angka | 9% |
| Kondisi bahaya → wajib eskalasi | 8% |

Tiga baris terakhir (25%) adalah alasan utama fine-tuning ini bernilai.

Teacher: **Gemini 3.1 Pro via Batch API**, `responseMimeType: application/json`
+ `responseSchema` sesuai §3.3. Prompt teacher memuat sel `decision_table` yang
relevan, rubrik keselamatan, larangan angka, dan register bahasa operator UMKM.

Filter otomatis (yang gagal **dibuang**, tidak diperbaiki):
skema valid, semua angka di whitelist, tidak ada frasa terlarang, panjang wajar.

Review manusia: 100% kasus CRITICAL, 100% kasus bahaya, 10% sampel acak sisanya.

Split **80/10/10 berdasarkan konteks**, bukan per giliran (cegah kebocoran).

Biaya perkiraan: ~1,6M token output → sekitar USD 10.

### 5.3 Fase 3 — LoRA (3 jam)

```
Unsloth di Colab T4
r=16, alpha=32, dropout=0.05
target: q,k,v,o,gate,up,down proj
3 epoch, lr 2e-4, cosine, bf16
max_seq_len 2048
train-on-completion-only (prompt di-mask)
checkpoint ke Drive tiap 100 step
```

Latih **dua ukuran** (pemenang Fase 0 + satu tingkat di bawahnya). Korpusnya sama,
biayanya hampir nol, hasilnya jadi tabel ablasi untuk proposal.

### 5.4 Fase 4 — Export (2 jam)

`merge_and_unload()` → `convert_hf_to_gguf.py` → `llama-quantize q4_k_m`
→ `Modelfile` (system prompt di-bake, temperature 0, top_k 1, seed 42)
→ `ollama create audiax-advisor`.

GGUF **wajib di-bake ke dalam image**, bukan di-pull saat runtime.

---

## 6. Fase 5 — Evaluasi (3 jam)

200 kasus dari test set yang tidak pernah dilihat saat training.

| Metrik | Gemma dasar | Gemma + LoRA | Target |
|---|---|---|---|
| Skema JSON valid | | | ≥ 98% |
| Pelanggaran guard angka | | | 0 |
| Frasa diagnosis terlarang | | | 0 |
| Safety-gate recall | | | 100% |
| Eskalasi benar pada kasus bahaya | | | 100% |
| Determinisme (2× run identik) | | | 200/200 |
| Latensi p95 (CPU) | | | < 5 dtk |
| Fallback rate | | | dilaporkan apa adanya |

Kolom "Gemma dasar" vs "Gemma + LoRA" adalah **bukti** fine-tuning. Tanpa itu,
"kami fine-tune" hanya klaim. Hasil ditulis ke `experiments/advisory/results.md`.

---

## 7. Definition of done per fase

| Fase | Selesai berarti |
|---|---|
| 0 | Tabel benchmark 3 model terisi + satu model dipilih + konversi GGUF terbukti jalan |
| 1a | 40 sel `decision_table.json` lengkap, lolos validasi skema, sudah di-review manusia |
| 1b | `rules_test.go` + `guard_test.go` hijau; `boundary_test.go` hijau |
| 1c | Endpoint balas 200 dengan `source: "fallback_static"` tanpa LLM apa pun |
| 2 | 3 file `.jsonl` (train/val/test) + catatan review manusia |
| 3 | 2 adapter LoRA + skor validasi keduanya |
| 4 | `docker compose up` menyala **tanpa internet**, `/api/.../messages` balas `source: "llm"` |
| 5 | `results.md` berisi tabel §6 terisi penuh |

---

## 8. Risiko

| Risiko | Mitigasi |
|---|---|
| Bahasa Indonesia model kecil buruk | Fase 0 mengukur sebelum investasi; fallback dev-time only |
| Latensi CPU di laptop juri | Fase 0 = gerbang go/no-go; batasi `num_predict` 220 |
| Konversi GGUF Gemma bermasalah | Diuji di Fase 0, bukan Fase 4 |
| Sesi Colab putus | Checkpoint ke Drive tiap 100 step |
| Korpus buatan LLM berkualitas rendah | Filter otomatis + gerbang review manusia |
| Train/serve skew pada format prompt | Format §3.2 adalah satu-satunya sumber kebenaran; Track A & B memakai file template yang sama |
| Tidak ada yang paham blower untuk review | Tulis konservatif: perbanyak "matikan & panggil teknisi" |
| Image Docker membengkak | q4_k_m + model terkecil yang lolos Fase 0 |
| Internet mati saat penjurian | Tidak berdampak: model lokal, GGUF di dalam image |

---

## 9. Yang harus ditulis di proposal

> Tim melatih dua model. Pertama, adapter akustik di atas BEATs beku untuk
> deteksi anomali suara mesin. Kedua, adapter bahasa (LoRA) di atas Gemma untuk
> menjelaskan hasil dan melayani tanya-jawab operator. Keduanya di-fine-tune
> sesuai fitur inovasi tim dan keduanya berjalan offline di dalam image Docker;
> tidak ada pemanggilan API model pihak ketiga saat runtime.
>
> Lapisan bahasa tidak memiliki kewenangan mengambil keputusan. Seluruh keputusan
> dihasilkan rule engine deterministik, dan setiap keluaran model diverifikasi
> validator yang menolak angka maupun klaim di luar fakta terukur.

Catatan penulisan: korpus disusun **dengan bantuan LLM lalu diverifikasi manusia**.
Hindari istilah "distilasi" tanpa kualifikasi domain-sempit.

---

## 10. Catatan yang belum selesai (di luar cakupan dokumen ini)

Kontradiksi compliance antar-repo — **sekarang terkonfirmasi dari kode**, bukan
lagi dugaan dari README:

| Bukti di `audiax_backend` | Klaim di repo AI / proposal |
|---|---|
| `db/migrations/0001..0004` — tabel `users`, `machines`, `baselines`, `inspections` | *"Dilarang infrastruktur database... jangan menambahkan database apa pun"* |
| `internal/delivery/http/middleware/auth.go`, sesi Redis, Bearer | *"MVP tanpa sistem otentikasi"* |
| `entity.Inspection` menyimpan status, z, health score per baris | *"Dilarang pipeline pencatatan data otomatis"* |
| `entity.Baseline` tersimpan di server | *"`MachineBaseline` disimpan di klien, bukan di server mana pun"* |

Ini **harus diputuskan tim**, dan keputusannya cuma dua: (a) revisi klaim di
proposal agar jujur menggambarkan sistem yang benar-benar dibangun, atau
(b) revisi backend agar sesuai klaim. Membiarkan keduanya berbeda adalah pilihan
terburuk — juri yang membuka repo akan menemukannya, dan itu menjatuhkan
kredibilitas seluruh bab compliance, bukan cuma butir yang salah.

Desain advisory ini tidak memperburuk keadaan: ia tidak menambah tabel baru dan
tidak menyimpan percakapan. Ia **memanfaatkan** tabel `inspections` yang sudah
ada untuk mengambil fakta klinis (§3.4) — karena alternatifnya, mempercayai
klien, adalah lubang keamanan.
