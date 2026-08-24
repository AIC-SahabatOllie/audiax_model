# Prompt pembuka — Sesi Backend (Track A)

Tempel isi blok di bawah ke sesi Claude Code yang dibuka di `C:\Code\audiax_backend`.

---

Kamu Track A pada fitur **Teknisi Saku** — lapisan advisory untuk AUDIAX.

## Baca dulu, sebelum menyentuh kode apa pun

1. `CLAUDE.md` repo ini
2. `C:\Code\audiax_model\experiments\advisory\DESIGN.md` — spec lengkap fitur ini, sudah disetujui
3. `internal/advisory/PROMPT_CONTRACT.md` — kontrak format prompt
4. `internal/advisory/prompt_template.txt` — template yang harus kamu render

DESIGN.md adalah sumber kebenaran. Kalau ada yang bertentangan antara instruksi
ini dan DESIGN.md, DESIGN.md yang menang — laporkan bedanya, jangan diam-diam
pilih salah satu.

## Konteks singkat

AUDIAX mendeteksi anomali suara mesin blower UMKM: BEATs + adapter menghasilkan
`HealthCard` (NORMAL / WARNING / CRITICAL / KALIBRASI_KURANG plus z-score).
Masalahnya, operator tidak tahu harus berbuat apa setelah melihat kartu itu.

Teknisi Saku menutup celah itu: tanya-jawab lanjutan yang dijelaskan model bahasa
lokal (Gemma hasil fine-tune, berjalan di container Ollama, offline).

**Aturan arsitektur yang tidak bisa ditawar:**
- **LLM tidak pernah memutuskan.** Rule engine yang memutuskan. LLM hanya menyusun kalimat di atas keputusan yang sudah dikunci.
- **LLM tidak pernah menghasilkan angka.** Semua angka berasal dari HealthCard dan tabel keputusan. Guard menolak output yang memuat angka di luar itu.
- **LLM tidak pernah mendiagnosis.** Sistem ini alat bantu triase, bukan classifier jenis kerusakan.
- Gagal guard atau LLM tidak tersedia ⇒ jatuh ke teks statis, **bukan** error. Fitur tidak pernah mati.

## Langkah pertama

Buat branch `feat/advisory` dari `main`. Repo sekarang ada di `feat/baseline`.

## Urutan pekerjaan

### Fase 1a — `decision_table.json` (~2 jam) ⛔ BERHENTI DI SINI

Tulis `internal/advisory/decision_table.json` sesuai DESIGN.md §3.1.

- Target ~40 sel, memakai wildcard `*`. Ruang penuh 4 dimensi = 192 kombinasi
  (`status|dominant_indicator|drive_type|recency`); wildcard menjaga tabel tetap
  bisa dibaca manusia satu per satu.
- Pencocokan: **wildcard paling sedikit menang.**
- Isi tiap sel **konservatif**. Tidak ada anggota tim yang pernah membongkar
  blower industri, jadi kalau ragu antara "periksa sendiri" dan "matikan dan
  panggil teknisi", pilih yang kedua. Checklist ini dibaca orang yang akan
  mendekati mesin berputar.
- Setiap `safety_gate` untuk status non-NORMAL wajib memerintahkan mematikan
  mesin dan menunggu impeler berhenti total.
- Bahasa Indonesia, register operator UMKM. Hindari istilah teknis tanpa penjelasan.

**Setelah selesai: LAPOR dan BERHENTI.** File ini adalah input Track B untuk
membuat korpus training. Mereka menunggu di jam ke-2. Jangan lanjut ke 1b sebelum
file ini beres dan sudah dibaca manusia.

### Fase 1b — inti modul (~5 jam)

```
internal/advisory/          <- PAKET MURNI
    rules.go                # lookup -> Cell, dengan aturan wildcard
    prompt.go               # //go:embed prompt_template.txt, render sesuai PROMPT_CONTRACT.md
    guard.go                # 4 validator, DESIGN.md §4.2
    provider.go             # interface LLMProvider { Complete(ctx, prompt) (string, error) }
    static.go               # fallback: susun balasan dari Cell tanpa LLM
    rules_test.go  prompt_test.go  guard_test.go  boundary_test.go
```

Empat guard, semua wajib lolos, gagal salah satu ⇒ fallback:
1. **Skema** — output parse sebagai JSON sesuai DESIGN.md §3.3
2. **Angka** — setiap angka di `jawaban` + `langkah_berikutnya` harus ada di
   whitelist (nilai dari HealthCard, dari Cell, ordinal 1–9)
3. **Diagnosis** — tolak frasa terlarang: `bearing aus`, `bearing rusak`,
   `impeler pecah`, `motor terbakar`, `kerusakan pada`, `disebabkan oleh`,
   `sisa umur`, `akan rusak dalam`
4. **Keselamatan** — bila `Cell.safety_gate` tidak kosong, output wajib memuat
   instruksi mematikan mesin sebelum instruksi memeriksa

`guard.go` adalah file terpenting di modul ini. Beri test paling banyak.
Ia harus tetap benar walaupun modelnya nanti diganti.

Dua test yang wajib ada di `rules_test.go`:
- setiap dari 192 kombinasi menemukan **tepat satu** sel
- tidak ada sel yang tidak pernah terpakai (dead cell)

### Fase 1c — lapisan HTTP (~1 jam)

```
internal/config/llmservice.go              # klien Ollama, meniru aiservice.go
internal/model/advisory.go                 # DTO request/response, DESIGN.md §3.4
internal/usecase/advisory.go               # orkestrasi + advisory_test.go
internal/delivery/http/advisory_controller.go
internal/delivery/http/route/route.go      # + 1 rute di setupAuthRoutes
internal/constants/constants.go            # + konstanta advisory
```

Rute:
```
POST /api/machines/:machineId/inspections/:inspectionId/advisory/messages
```

## Konvensi repo — dibaca dari kode, bukan asumsi

| Hal | Aturan |
|---|---|
| Framework | Fiber v2. Module `audiax`, Go 1.25 |
| Konstanta | **Semua** ke `internal/constants/constants.go`. Package doc-nya eksplisit: *"If you are about to write `const` anywhere else, write it here instead."* Jadi kata kunci bahaya, path Ollama, timeout — semua ke sana. Jangan buat `danger_keywords.go` |
| Klien HTTP keluar | Tiru `internal/config/aiservice.go` persis. Klien Ollama ke `internal/config/llmservice.go` |
| Error domain | `internal/apperr` — `ErrUnavailable`, `RejectedError` |
| Test | `*_test.go` bersebelahan, `testify`, fake di `fakes_test.go` |
| Layering | delivery → usecase → repository. Usecase tidak tahu HTTP |
| Konstanta yang sudah ada | Pakai ulang `StatusNormal`/`StatusWarning`/`StatusCritical`/`StatusUncalibrated`, `TriageDisclaimer`, `CalibrationQualityGood`. Jangan tulis ulang literalnya |

Konstanta baru yang perlu ditambah:
```go
OllamaChatPath          = "/api/chat"
OllamaModelName         = "audiax-advisor"
DefaultAdvisoryTimeout  = 5 * time.Second   // kontras dengan DefaultAITimeout 240s
AdvisoryMaxHistoryTurns = 8
AdvisoryNumPredict      = 220
var DangerKeywords = []string{...}          // DESIGN.md §3.5
```

## Empat jebakan yang sudah teridentifikasi — jangan terperosok

1. **Fakta klinis diambil dari DB, BUKAN dari klien.** Ambil baris `inspections`
   lewat `:inspectionId`, cek kepemilikan via `machines.user_id`. Kalau klien
   boleh mengirim `status`, klien bisa memalsukan `NORMAL` pada mesin CRITICAL
   dan seluruh rantai keselamatan runtuh.

2. **`calibration_quality` tidak ada di tabel `inspections`.** Cek
   `internal/entity/inspection.go` — memang tidak ada kolomnya. Join dari
   `baselines` lewat `inspections.baseline_id`. Kalau join gagal, hilangkan
   barisnya dari prompt; jangan tulis `null`.

3. **Atribut mesin (drive_type, umur, jam operasi, cadangan) belum punya kolom.**
   `entity.Machine` cuma punya label/location/description. Untuk sekarang
   dikirim klien di field `context` — sengaja, supaya fitur ini tidak terblokir
   perubahan CRUD mesin. Migrasi `0005_add_machine_attributes` masuk daftar
   lanjutan, **jangan** dikerjakan sekarang.

4. **Kata kunci bahaya dicek deterministik di Go, sebelum LLM dipanggil.**
   Cocok ⇒ paksa jalur CRITICAL, `escalated=true`, `needs_technician=true`,
   gerbang keselamatan dipasang, apa pun status akustiknya. LLM tidak dilibatkan
   dalam keputusan itu. Field `eskalasi` dari LLM hanya saran, bukan penentu.

## `boundary_test.go`

`internal/advisory/` **dilarang** meng-import: GORM, `internal/entity`,
`internal/config`, `internal/repository`, `internal/usecase`, Fiber.
**Diizinkan:** pustaka standar dan `internal/constants` (paket itu tidak
meng-import apa pun dari modul ini, jadi tidak bisa menyeret dependensi masuk).

Test memindai import dan gagal bila dilanggar — pola yang sama dengan
`tests/test_boundary.py` di repo AI.

## Jangan

- Jangan sentuh repo `C:\Code\audiax_model` (baca boleh, tulis tidak)
- Jangan ubah `prompt_template.txt`. Kalau formatnya terasa salah, **lapor** —
  mengubahnya setelah korpus digenerate membuat korpus itu basi
- Jangan tambah tabel atau migrasi baru
- Jangan tambah dependensi Go baru tanpa alasan kuat; Ollama dipanggil dengan
  `net/http` biasa, persis seperti `aiservice.go`
- Jangan menulis angka hasil pengukuran yang belum pernah benar-benar diukur
- Jangan buat komponen baru tanpa test

## Definition of done

| Fase | Selesai berarti |
|---|---|
| 1a | ~40 sel lengkap, lolos validasi skema, sudah dibaca manusia |
| 1b | `rules_test.go`, `prompt_test.go`, `guard_test.go`, `boundary_test.go` hijau |
| 1c | `make test` hijau; endpoint balas 200 dengan `source: "fallback_static"` tanpa LLM apa pun berjalan |

Setelah 1c, fitur sudah berfungsi penuh **tanpa LLM sama sekali**. Integrasi
Ollama menyusul setelah Track B selesai melatih modelnya.

Mulai dari Fase 1a. Laporkan dan berhenti setelah `decision_table.json` beres.
