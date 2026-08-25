"""Fase 2 — pembangkit korpus advisory dari Gemini sebagai teacher.

Mengimplementasikan ``CORPUS_SPEC.md``. Kalau keduanya berbeda, dokumen itu yang
menang dan skrip ini yang salah.

Alur: sampel konteks bertingkat -> lookup sel decision_table -> render prompt
memakai template yang SAMA dengan produksi -> minta Gemini menulis jawaban
teladan -> filter otomatis -> tulis train/val/test.

Pustaka standar saja (mesin dev memakai Python 3.14 yang belum tentu punya wheel
untuk paket pihak ketiga). API Gemini dipanggil lewat REST, bukan SDK.

Pakai:
    export GEMINI_API_KEY=...
    python gen_corpus.py --contexts 600
    python gen_corpus.py --contexts 12 --dry-run    # tanpa memanggil API
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple

BACKEND = Path(r"C:\Code\audiax_backend\internal\advisory")
TABLE_PATH = BACKEND / "decision_table.json"
TEMPLATE_PATH = BACKEND / "prompt_template.txt"
OUT_DIR = Path(__file__).with_name("corpus")

# Teacher bertingkat. Tugas ini bukan penalaran berat -- ia mengikuti aturan
# ketat dan menulis Indonesia yang wajar -- jadi Flash cukup untuk mayoritas,
# dan pada 2.000 panggilan selisih kecepatannya berarti satu jam vs dua setengah.
# Kasus yang salahnya paling mahal tetap dapat model terkuat.
# Diukur terhadap kunci yang dipakai, bukan diasumsikan dari daftar model:
#   gemini-3.1-pro-preview  -> 429, berbayar-saja, tidak tersedia di tier ini
#   gemini-3.7-flash        -> 503, model sedang kelebihan beban
#   gemini-3.6-flash        -> 200  <- dipakai
#   gemini-3.5-flash        -> 200  <- cadangan saat 503
# Rencana teacher bertingkat (Pro untuk kasus keselamatan) gugur karena Pro
# tidak bisa diakses; kompensasinya ada di filter otomatis dan gerbang review.
# Tier GRATIS. Tidak ada tagihan yang mungkin masuk: kuota habis menghasilkan
# HTTP 429, bukan biaya. Model berbayar-saja (gemini-3.1-pro-preview) sengaja
# tidak pernah dipanggil.
#
# flash-lite dipilih bukan cuma karena termurah, tapi karena rate limit tier
# gratisnya paling longgar -- dan di sini yang mengikat adalah RPM, bukan uang.
MODEL_DEFAULT = "gemini-3.5-flash-lite"
# Diukur, bukan diasumsikan: 3.6-flash ter-throttle berat pada kunci ini dan
# tiap panggilannya membakar retry backoff sampai ~190 detik. Karena teacher
# bertingkat merutekan ~40% trafik ke sana (semua CRITICAL + 3 niat
# keselamatan), batch 5 konteks pun tidak selesai dalam 5 menit.
# flash-lite: 12/12 sukses, 51 panggilan/menit, nol 429. Dipakai untuk semua.
MODEL_CRITICAL = "gemini-3.5-flash-lite"
CRITICAL_INTENTS = {"bahaya", "pancingan_angka", "di_luar_cakupan"}

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={k}"


def teacher_for(intent: str, status: str) -> str:
    if intent in CRITICAL_INTENTS or status == "CRITICAL":
        return MODEL_CRITICAL
    return MODEL_DEFAULT

# ---------------------------------------------------------------------------
# Ruang konteks (CORPUS_SPEC §1)
# ---------------------------------------------------------------------------

DIMS: Dict[str, List[Any]] = {
    "status": ["NORMAL", "WARNING", "CRITICAL", "KALIBRASI_KURANG"],
    "dominant_indicator": ["kurtosis", "crest_factor", "spectral_centroid", None],
    "drive_type": ["belt", "direct-coupled", "direct-drive"],
    "recency": ["<1bln", "1-6bln", ">6bln", "tidak-tahu"],
    "machine_age": ["<1th", "1-3th", "3-5th", ">5th"],
    "hours_per_day": ["<4", "4-8", ">8"],
    "has_backup": [True, False],
    "load_state": ["kosong", "bermuatan"],
}

# Porsi menurut KEPENTINGAN, bukan frekuensi alami. CRITICAL jarang di lapangan
# tapi paling mahal kalau jawabannya salah, jadi porsinya dinaikkan.
STATUS_MIX = {"NORMAL": 0.20, "WARNING": 0.40, "CRITICAL": 0.30, "KALIBRASI_KURANG": 0.10}

INTENT_MIX = {
    "istilah": 0.15,
    "boleh_jalan": 0.15,
    "observasi_baru": 0.20,
    "lanjutan": 0.15,
    "teknisi_biaya": 0.10,
    "di_luar_cakupan": 0.08,
    "pancingan_angka": 0.09,
    "bahaya": 0.08,
}

INTENT_QUESTIONS: Dict[str, List[str]] = {
    "istilah": [
        "{indicator} itu apa sih?",
        "maksudnya angka {z} itu apa ya?",
        "saya nggak ngerti istilahnya, tolong pakai bahasa gampang",
    ],
    "boleh_jalan": [
        "hari ini masih aman dipakai nggak?",
        "boleh saya nyalakan buat batch pagi?",
        "kalau dipakai setengah hari aja gimana?",
    ],
    "observasi_baru": [
        "ada bunyi tok-tok halus yang tadinya nggak ada",
        "kok sekarang mesinnya lebih berisik dari kemarin",
        "belakangan ini agak bergetar kalau lagi penuh",
        "suaranya mendengung terus nggak berhenti",
    ],
    "lanjutan": [
        "udah saya cek semua, normal semua. terus gimana?",
        "sabuknya udah saya kencengin, lanjut apa lagi?",
        "bautnya udah saya kencengin tapi suaranya masih sama",
    ],
    "teknisi_biaya": [
        "ini perlu panggil teknisi nggak?",
        "kira-kira benerinnya mahal nggak ya?",
        "teknisinya harus datang hari ini juga?",
    ],
    "di_luar_cakupan": [
        "harga blower baru berapa ya?",
        "merek mesin apa yang bagus buat oven kerupuk?",
        "bisa bantu carikan teknisi di daerah saya?",
        "gimana cara bikin kerupuk biar lebih renyah?",
    ],
    "pancingan_angka": [
        "kira-kira berapa hari lagi mesin ini rusak?",
        "akurasi sistem ini berapa persen?",
        "umur mesin saya tinggal berapa bulan?",
        "peluang rusaknya berapa persen?",
    ],
    "bahaya": [
        "tadi ada bau gosong dikit dari motornya",
        "keluar asap tipis waktu dinyalakan",
        "rumah bearingnya panas banget sampai nggak bisa dipegang",
        "getarannya keras sekali sampai mejanya ikut goyang",
    ],
}

INDICATOR_LABEL = {
    "kurtosis": "kurtosis",
    "crest_factor": "crest factor",
    "spectral_centroid": "spectral centroid",
    None: "indikatornya",
}

DRIVE_LABEL = {"belt": "sabuk-puli", "direct-coupled": "kopling langsung", "direct-drive": "direct-drive"}
RECENCY_LABEL = {"<1bln": "<1 bulan", "1-6bln": "1-6 bulan", ">6bln": ">6 bulan", "tidak-tahu": "tidak tahu"}
AGE_LABEL = {"<1th": "<1 tahun", "1-3th": "1-3 tahun", "3-5th": "3-5 tahun", ">5th": ">5 tahun"}

Z_BY_STATUS = {"NORMAL": (0.2, 2.8), "WARNING": (3.0, 5.9), "CRITICAL": (6.0, 11.0), "KALIBRASI_KURANG": None}

# ---------------------------------------------------------------------------
# Lookup decision_table (aturan: wildcard paling sedikit menang)
# ---------------------------------------------------------------------------


def load_table() -> List[Dict[str, Any]]:
    data = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    return data["cells"] if isinstance(data, dict) else data


def lookup(cells: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    want = [
        str(ctx["status"]),
        str(ctx["dominant_indicator"]) if ctx["dominant_indicator"] else "null",
        str(ctx["drive_type"]),
        str(ctx["recency"]),
    ]
    best, best_wild = None, 99
    for cell in cells:
        parts = cell["key"].split("|")
        if len(parts) != 4:
            continue
        wild = 0
        ok = True
        for got, exp in zip(want, parts):
            if exp == "*":
                wild += 1
            elif exp != got:
                ok = False
                break
        if ok and wild < best_wild:
            best, best_wild = cell, wild
    return best


# ---------------------------------------------------------------------------
# Render prompt -- WAJIB identik dengan prompt.go (PROMPT_CONTRACT.md)
# ---------------------------------------------------------------------------


def render_prompt(template: str, f: Dict[str, Any]) -> str:
    v = dict(f)
    z = v.get("z")
    v["z"] = "" if z is None else f"{float(z):.1f}"
    b = v.get("has_backup")
    v["has_backup"] = "" if b is None else ("ya" if b else "tidak")
    v["checklist"] = "\n".join(f"    {i}. {s}" for i, s in enumerate(v.pop("checklist", []) or [], 1))
    hist = v.pop("history", []) or []
    v["history"] = "\n".join(
        f"  {'operator' if t['role'] == 'user' else 'asisten'}: {t['content']}" for t in hist[-8:]
    )

    out: List[str] = []
    for line in template.splitlines():
        rendered, drop = line, False
        for key, val in v.items():
            tok = "{" + key + "}"
            if tok not in rendered:
                continue
            text = "" if val is None else str(val)
            if text == "":
                drop = True
                break
            rendered = rendered.replace(tok, text)
        if not drop:
            out.append(rendered.rstrip())
    if not v["history"]:
        out = [l for l in out if l.strip() != "RIWAYAT:"]

    # Samakan dengan collapseBlankRuns() di prompt.go. Menghapus blok RIWAYAT:
    # menyisakan dua baris kosong berurutan (pemisah sebelum dan sesudahnya),
    # sementara Go menggabungkannya jadi satu. Tanpa ini, model dilatih pada
    # format yang berbeda satu baris kosong dari yang dilihatnya di produksi --
    # train/serve skew yang gejalanya halus dan mahal dilacak.
    collapsed: List[str] = []
    for line in out:
        if not line.strip() and collapsed and not collapsed[-1].strip():
            continue
        collapsed.append(line)
    return "\n".join(collapsed) + "\n"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_contexts(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for status, share in STATUS_MIX.items():
        for _ in range(max(1, round(n * share))):
            ind = None if status in ("NORMAL", "KALIBRASI_KURANG") else rng.choice(DIMS["dominant_indicator"][:3])
            zr = Z_BY_STATUS[status]
            ctx = {
                "status": status,
                "dominant_indicator": ind,
                "drive_type": rng.choice(DIMS["drive_type"]),
                "recency": rng.choice(DIMS["recency"]),
                "machine_age": rng.choice(DIMS["machine_age"]),
                "hours_per_day": rng.choice(DIMS["hours_per_day"]),
                # CRITICAL tanpa mesin cadangan = konsekuensi terberat, diperbanyak.
                "has_backup": rng.random() > (0.7 if status == "CRITICAL" else 0.5),
                "load_state": rng.choice(DIMS["load_state"]),
                "z": None if zr is None else round(rng.uniform(*zr), 1),
            }
            out.append(ctx)
    rng.shuffle(out)
    return out[:n]


def pick_intents(rng: random.Random, k: int, focus: Optional[List[str]] = None) -> List[str]:
    """Pilih niat menurut INTENT_MIX, atau merata atas ``focus`` kalau diberikan.

    Mode fokus ada karena niat yang paling penting justru paling jarang muncul
    di sampling proporsional: ``pancingan_angka`` dan ``di_luar_cakupan``
    masing-masing hanya 9% dan 8%. Padahal itu dua perilaku penolakan yang
    paling sering gagal di model dasar, dan CORPUS_SPEC mensyaratkan minimal 20
    kasus per niat di test set. Menambah porsinya lewat batch terarah jauh lebih
    murah daripada menggenerate ulang seluruh korpus.
    """
    if focus:
        return [rng.choice(focus) for _ in range(k)]
    names = list(INTENT_MIX)
    weights = [INTENT_MIX[n] for n in names]
    return rng.choices(names, weights=weights, k=k)


# ---------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------

TEACHER_SYSTEM = """Kamu menulis contoh jawaban untuk asisten pemeliharaan mesin blower yang dipakai operator UMKM pangan kering di Indonesia.

PEMBACA: operator pabrik kecil, bukan teknisi. Sebagian tidak tamat SMA. Bahasa Indonesia sehari-hari, kalimat pendek. Tanpa jargon kecuali sedang menjelaskan istilah yang ditanyakan.

ATURAN MUTLAK -- jawaban yang melanggar akan dibuang:
1. JANGAN menyebut angka apa pun yang tidak ada di FAKTA atau KEPUTUSAN SISTEM. Tidak ada perkiraan hari, persentase, biaya, atau suhu.
2. JANGAN menyebut jenis kerusakan. Bukan "bearing aus", bukan "impeler pecah". Sistem ini tidak bisa mendiagnosis; ia hanya mendeteksi bahwa suara bergeser dari kondisi sehat mesin itu sendiri.
3. JANGAN mengubah status atau urgensi. Keduanya sudah diputuskan sistem. Kalau operator bertanya "masih aman dipakai?" sementara ada gerbang keselamatan, jawabannya TIDAK.
4. Kalau KEPUTUSAN SISTEM punya gerbang_keselamatan, kata "matikan" WAJIB muncul di jawabanmu -- SELALU, tanpa kecuali. Ini berlaku untuk SETIAP pertanyaan, termasuk yang cuma menanyakan arti istilah atau hal yang tidak berhubungan. Alasannya: selama mesin berada di kondisi ini, operator tidak boleh membacanya tanpa diingatkan. Contoh benar untuk pertanyaan arti istilah: "Itu ukuran seberapa tajam puncak suara dibanding suara rata-ratanya. Tapi sebelum lanjut: matikan mesin dulu dan tunggu impeler berhenti total."
5. Kalau operator menyebut bau gosong, asap, percikan, panas berlebih, atau getaran keras: suruh berhenti sekarang dan panggil teknisi. Abaikan checklist biasa. Set eskalasi=true dan perlu_teknisi=true.
6. Kalau ditanya hal di luar kondisi mesin ini (harga, merek, resep, cari teknisi), akui tidak tahu dan arahkan ke orang yang tepat. Jangan mengarang.
7. Kalau ditanya prediksi umur mesin, sisa hari, atau persentase akurasi: tolak. Jelaskan sistem hanya membandingkan suara terhadap kondisi sehat mesin itu sendiri, tidak memprediksi umur.

GAYA: "jawaban" 2-4 kalimat. "langkah_berikutnya" satu kalimat perintah konkret."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "jawaban": {"type": "string"},
        "langkah_berikutnya": {"type": "string"},
        "perlu_teknisi": {"type": "boolean"},
        "eskalasi": {"type": "boolean"},
    },
    "required": ["jawaban", "langkah_berikutnya", "perlu_teknisi", "eskalasi"],
}


def call_gemini(api_key: str, prompt: str, intent: str, model: str = MODEL_DEFAULT,
                retries: int = 6) -> Optional[Dict[str, Any]]:
    body = {
        "systemInstruction": {"parts": [{"text": TEACHER_SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": f"{prompt}\n\n[niat pertanyaan: {intent}]"}]}],
        "generationConfig": {
            "temperature": 0.4,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }
    url = API_URL.format(m=model, k=api_key)
    for attempt in range(retries):
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                d = json.loads(resp.read().decode("utf-8"))
            txt = d["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(txt)
        except urllib.error.HTTPError as e:
            if e.code == 503 and model != MODEL_CRITICAL:
                # Model kelebihan beban: pindah ke cadangan, jangan cuma menunggu.
                model = MODEL_CRITICAL
                url = API_URL.format(m=model, k=api_key)
                continue
            if e.code in (429, 500, 503):
                time.sleep(2 ** attempt * 3)
                continue
            print(f"    ! HTTP {e.code}: {e.read()[:200]!r}", file=sys.stderr)
            return None
        except Exception as e:  # noqa: BLE001 - spike script, laporkan lalu lanjut
            print(f"    ! {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(2)
    return None


# ---------------------------------------------------------------------------
# Filter otomatis (CORPUS_SPEC §5) -- yang gagal DIBUANG, bukan diperbaiki
# ---------------------------------------------------------------------------

BANNED = [
    "bearing aus", "bearing rusak", "impeler pecah", "motor terbakar",
    "kerusakan pada", "disebabkan oleh", "sisa umur", "akan rusak dalam",
]
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def allowed_numbers(ctx: Dict[str, Any], cell: Dict[str, Any]) -> set:
    ok = {str(i) for i in range(1, 10)}
    if ctx.get("z") is not None:
        ok |= {f"{ctx['z']:.1f}", f"{ctx['z']:.1f}".replace(".", ",")}
    ok |= {"3.0", "6.0", "3,0", "6,0", "3", "6"}
    ok.add(str(cell.get("recheck_hours", "")))
    for step in cell.get("checklist", []):
        ok |= set(NUM_RE.findall(step))
    ok |= set(NUM_RE.findall(cell.get("safety_gate", "") or ""))
    ok |= set(NUM_RE.findall(cell.get("escalate_if", "") or ""))
    return ok


def check(ans: Dict[str, Any], ctx: Dict[str, Any], cell: Dict[str, Any], intent: str) -> Optional[str]:
    """Kembalikan alasan gagal, atau None kalau lolos."""
    for k in ("jawaban", "langkah_berikutnya"):
        if not isinstance(ans.get(k), str):
            return f"skema: {k} bukan string"
    for k in ("perlu_teknisi", "eskalasi"):
        if not isinstance(ans.get(k), bool):
            return f"skema: {k} bukan boolean"

    text = f"{ans['jawaban']} {ans['langkah_berikutnya']}"
    low = text.lower()

    for b in BANNED:
        if b in low:
            return f"diagnosis: '{b}'"

    ok = allowed_numbers(ctx, cell)
    for n in NUM_RE.findall(text):
        if n not in ok:
            return f"angka asing: '{n}'"

    gate = (cell.get("safety_gate") or "").strip()
    if gate and "matikan" not in low:
        return "gerbang keselamatan hilang"

    if intent == "bahaya" and not (ans["eskalasi"] and ans["perlu_teknisi"]):
        return "kasus bahaya tidak dieskalasi"

    if not (20 <= len(ans["jawaban"]) <= 400):
        return f"panjang jawaban {len(ans['jawaban'])}"
    if not (10 <= len(ans["langkah_berikutnya"]) <= 200):
        return f"panjang langkah {len(ans['langkah_berikutnya'])}"

    return None


# ---------------------------------------------------------------------------


def build_facts(ctx: Dict[str, Any], cell: Dict[str, Any], question: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "status": ctx["status"],
        "z": ctx["z"],
        "z_warning": "3.0",
        "z_critical": "6.0",
        "dominant_indicator": ctx["dominant_indicator"],
        "calibration_quality": "rendah" if ctx["status"] == "KALIBRASI_KURANG" else "baik",
        "drive_type": DRIVE_LABEL[ctx["drive_type"]],
        "recency": RECENCY_LABEL[ctx["recency"]],
        "machine_age": AGE_LABEL[ctx["machine_age"]],
        "hours_per_day": ctx["hours_per_day"],
        "has_backup": ctx["has_backup"],
        "load_state": ctx["load_state"],
        "urgency": cell["urgency"],
        "safety_gate": cell.get("safety_gate", ""),
        "checklist": cell.get("checklist", []),
        "escalate_if": cell.get("escalate_if", ""),
        "recheck_hours": cell.get("recheck_hours", ""),
        "history": history,
        "user_message": question,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, default=600)
    ap.add_argument("--turns-per-context", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--focus", type=str, default="", help="niat yang difokuskan, dipisah koma")
    ap.add_argument("--workers", type=int, default=3, help="konteks diproses paralel")
    ap.add_argument("--dry-run", action="store_true", help="jangan panggil API, cetak contoh prompt")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key and not args.dry_run:
        print("GEMINI_API_KEY belum disetel.", file=sys.stderr)
        return 2

    cells = load_table()
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    tpl_sha = hashlib.sha256(template.encode("utf-8")).hexdigest()
    tbl_sha = hashlib.sha256(TABLE_PATH.read_bytes()).hexdigest()
    rng = random.Random(args.seed)

    print(f"sel decision_table : {len(cells)}")
    print(f"template sha256    : {tpl_sha[:16]}...")
    print(f"tabel sha256       : {tbl_sha[:16]}...")

    contexts = sample_contexts(args.contexts, rng)
    records: List[Dict[str, Any]] = []
    dropped: Dict[str, int] = {}
    seen_intent: Dict[str, int] = {}

    # Giliran DI DALAM satu konteks harus berurutan -- riwayat percakapan
    # dibangun dari jawaban giliran sebelumnya. Antar-konteks tidak saling
    # bergantung, jadi di situlah paralelismenya. Pada 600 konteks ini selisih
    # sekitar satu jam versus sepuluh menit.
    def run_context(item: Tuple[int, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        ci, ctx = item
        got: List[Dict[str, Any]] = []
        bad: List[str] = []
        intents_done: List[str] = []

        cell = lookup(cells, ctx)
        if cell is None:
            return got, ["tidak ada sel cocok"], intents_done

        # RNG sendiri per konteks: rng bersama tidak thread-safe, dan yang lebih
        # penting, hasilnya jadi tidak reproducible antar-jalan.
        local = random.Random(args.seed * 1_000_003 + ci)
        history: List[Dict[str, str]] = []

        focus = [x.strip() for x in args.focus.split(',') if x.strip()] or None
        for intent in pick_intents(local, args.turns_per_context, focus):
            q = local.choice(INTENT_QUESTIONS[intent])
            q = q.replace("{indicator}", INDICATOR_LABEL[ctx["dominant_indicator"]])
            q = q.replace("{z}", str(ctx["z"]))
            prompt = render_prompt(template, build_facts(ctx, cell, q, history))

            if args.dry_run:
                if ci == 0 and not intents_done:
                    print("\n--- CONTOH PROMPT ---\n" + prompt + "---\n")
                intents_done.append(intent)
                continue

            ans = call_gemini(api_key, prompt, intent, teacher_for(intent, ctx["status"]))
            if ans is None:
                bad.append("api gagal")
                continue
            why = check(ans, ctx, cell, intent)
            if why:
                bad.append(why.split(":")[0])
                continue

            rec = {
                # Diberi awalan seed karena korpus dibangun dari beberapa batch
                # yang di-append ke raw.jsonl. Tanpa ini, konteks ke-3 dari batch
                # A dan batch B dianggap konteks yang sama, dan pembagian
                # train/test jadi bocor antar-batch.
                "context_id": f"{args.seed}-{ci}",
                "intent": intent,
                "status": ctx["status"],
                "prompt": prompt,
                "completion": json.dumps(ans, ensure_ascii=False),
            }
            got.append(rec)
            # Tulis SEGERA. Job 600 konteks berjalan ~20 menit dan sudah pernah
            # dihentikan di tengah; tanpa ini seluruh kemajuan hangus. raw.jsonl
            # adalah sumber kebenaran, train/val/test cuma turunannya.
            with _raw_lock:
                _raw_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                _raw_fh.flush()
            history = history + [
                {"role": "user", "content": q},
                {"role": "assistant", "content": ans["jawaban"]},
            ]
            intents_done.append(intent)

        return got, bad, intents_done

    OUT_DIR.mkdir(exist_ok=True)
    raw_path = OUT_DIR / "raw.jsonl"
    global _raw_fh, _raw_lock
    _raw_lock = threading.Lock()
    # Mode append: menjalankan ulang dengan --seed berbeda menambah korpus,
    # tidak menimpanya.
    _raw_fh = raw_path.open("a", encoding="utf-8")

    items = list(enumerate(contexts))
    if args.dry_run or args.workers <= 1:
        results = [run_context(it) for it in items]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = []
            for i, res in enumerate(pool.map(run_context, items), 1):
                results.append(res)
                if i % 5 == 0 or i == len(items):
                    kept = sum(len(r[0]) for r in results)
                    print(f"  {i}/{len(items)} konteks | {kept} giliran lolos", flush=True)

    for got, bad, intents_done in results:
        records.extend(got)
        for b in bad:
            dropped[b] = dropped.get(b, 0) + 1
        for it in intents_done:
            seen_intent[it] = seen_intent.get(it, 0) + 1

    if args.dry_run:
        print("distribusi niat (dry run):", dict(sorted(seen_intent.items())))
        return 0

    _raw_fh.close()
    records = [json.loads(l) for l in raw_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"total record di raw.jsonl: {len(records)}")

    by_ctx: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        by_ctx.setdefault(r["context_id"], []).append(r)
    ids = sorted(by_ctx, key=str)
    rng.shuffle(ids)
    n = len(ids)
    splits = {
        "train": ids[: int(n * 0.8)],
        "val": ids[int(n * 0.8): int(n * 0.9)],
        "test": ids[int(n * 0.9):],
    }
    for name, keep in splits.items():
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for cid in keep:
                for r in by_ctx[cid]:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {sum(len(by_ctx[c]) for c in keep)} giliran / {len(keep)} konteks -> {path.name}")

    (OUT_DIR / "meta.json").write_text(
        json.dumps({
            "teacher_model": {"default": MODEL_DEFAULT, "critical": MODEL_CRITICAL},
            "prompt_template_sha256": tpl_sha,
            "decision_table_sha256": tbl_sha,
            "n_contexts": len(contexts),
            "n_turns_kept": len(records),
            "dropped_by_reason": dropped,
            "intent_counts": seen_intent,
            "seed": args.seed,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\ndibuang:", dropped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
