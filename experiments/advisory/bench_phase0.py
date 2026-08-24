"""Fase 0 — spike benchmark kandidat base model untuk lapisan advisory.

Tujuan: gerbang go/no-go. Mengukur apakah model kecil yang berjalan di CPU
dengan batas resource seperti di ``docker-compose.yml`` (4 CPU) bisa menjawab
dalam waktu yang masih layak untuk operator yang sedang menunggu.

Ini SPIKE, bukan kode produksi: hasilnya angka, bukan artefak yang dipertahankan.
Tapi fungsi ``render_prompt`` di sini sengaja ditulis sesuai
``audiax_backend/internal/advisory/PROMPT_CONTRACT.md`` karena ia akan dipakai
ulang sebagai dasar pembuat korpus di Fase 2 -- dan train/serve skew di sana
adalah risiko nomor satu proyek ini.

Pustaka standar saja: mesin ini memakai Python 3.14 yang belum tentu punya wheel
untuk paket pihak ketiga.

Pakai:
    python bench_phase0.py gemma3:270m gemma3:1b
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

OLLAMA_URL = "http://localhost:11434/api/chat"

#: Harus sama dengan batas ``cpus: '4.0'`` di docker-compose.yml. Mengukur dengan
#: 12 thread akan memberi angka 2-3x lebih optimistis dari kondisi nyata.
NUM_THREAD = 4
NUM_PREDICT = 220

TEMPLATE_PATH = Path(
    r"C:\Code\audiax_backend\internal\advisory\prompt_template.txt"
)


# ---------------------------------------------------------------------------
# Render prompt -- aturan dari PROMPT_CONTRACT.md
# ---------------------------------------------------------------------------


def render_prompt(template: str, facts: Dict[str, Any]) -> str:
    """Render template sesuai aturan PROMPT_CONTRACT.md.

    Aturan yang ditegakkan di sini (harus identik dengan sisi Go):
      1. ``{placeholder}`` diganti nilainya apa adanya.
      2. Baris yang placeholder-nya kosong DIHAPUS seluruhnya -- bukan ditulis
         ``null``, bukan disisakan baris kosong.
      3. ``{checklist}`` jadi baris bernomor, indentasi 4 spasi.
      4. ``{history}`` maksimal 8 giliran, indentasi 2 spasi. Riwayat kosong
         menghapus seluruh blok termasuk header ``RIWAYAT:``.
      5. ``has_backup`` dirender ``ya``/``tidak``, bukan ``true``/``false``.
      6. ``z`` satu angka di belakang koma; ``None`` menghapus barisnya.
    """
    values = dict(facts)

    z = values.get("z")
    values["z"] = "" if z is None else f"{float(z):.1f}"

    backup = values.get("has_backup")
    values["has_backup"] = "" if backup is None else ("ya" if backup else "tidak")

    checklist = values.pop("checklist", []) or []
    values["checklist"] = "\n".join(
        f"    {i}. {step}" for i, step in enumerate(checklist, start=1)
    )

    history = values.pop("history", []) or []
    values["history"] = "\n".join(
        f"  {'operator' if turn['role'] == 'user' else 'asisten'}: {turn['content']}"
        for turn in history[-8:]
    )

    out_lines: List[str] = []
    for line in template.splitlines():
        rendered = line
        drop = False
        for key, val in values.items():
            token = "{" + key + "}"
            if token not in rendered:
                continue
            text = "" if val is None else str(val)
            if text == "":
                drop = True
                break
            rendered = rendered.replace(token, text)
        if drop:
            continue
        out_lines.append(rendered.rstrip())

    # Header RIWAYAT yang tidak diikuti isi apa pun ikut dibuang.
    if not values["history"]:
        out_lines = [ln for ln in out_lines if ln.strip() != "RIWAYAT:"]

    return "\n".join(out_lines) + "\n"


# ---------------------------------------------------------------------------
# Fixture -- sengaja hardcoded
# ---------------------------------------------------------------------------

# decision_table.json belum ada saat spike ini ditulis (Track A sedang
# mengerjakannya). Tiga sel di bawah adalah tiruan yang cukup realistis untuk
# mengukur panjang prompt dan waktu inferensi. GANTI dengan sel asli begitu
# tabelnya tersedia -- panjang prompt memengaruhi latensi.
CELL_WARNING_BELT = {
    "urgency": "rencanakan dalam 48 jam",
    "safety_gate": "Matikan mesin dan tunggu impeler berhenti total sebelum memeriksa apa pun.",
    "checklist": [
        "Periksa ketegangan sabuk - tekan di tengah, lendutan wajar sekitar 1 cm per 100 cm jarak puli.",
        "Lihat permukaan sabuk: retak, mengkilap, atau serat terkelupas.",
        "Raba rumah bearing saat mesin mati: terasa jauh lebih panas dari biasanya?",
        "Cek baut dudukan motor dan blower, kencangkan yang longgar.",
    ],
    "escalate_if": "Kalau keempat hal di atas normal tapi besok statusnya tetap WARNING, panggil teknisi.",
    "recheck_hours": 48,
}

CELL_NORMAL = {
    "urgency": "tidak ada tindakan mendesak",
    "safety_gate": "",
    "checklist": ["Lanjutkan pemakaian seperti biasa."],
    "escalate_if": "Kalau muncul bunyi atau bau yang tidak biasa, periksa lebih awal.",
    "recheck_hours": 168,
}

CELL_CRITICAL = {
    "urgency": "hentikan sekarang",
    "safety_gate": "Matikan mesin sekarang dan jangan dinyalakan lagi sampai diperiksa teknisi.",
    "checklist": [
        "Matikan mesin dari sakelar utama.",
        "Jangan buka penutup apa pun selagi impeler masih berputar.",
        "Hubungi teknisi hari ini juga.",
    ],
    "escalate_if": "Kalau teknisi belum bisa datang, jangan jalankan mesin untuk sementara.",
    "recheck_hours": 0,
}

BASE_FACTS = {
    "z_warning": "3.0",
    "z_critical": "6.0",
    "machine_age": "3-5 tahun",
    "hours_per_day": ">8",
    "load_state": "bermuatan",
}


def _case(name, status, z, indicator, cell, question, history=None, **over):
    facts = dict(BASE_FACTS)
    facts.update(
        status=status,
        z=z,
        dominant_indicator=indicator,
        calibration_quality="baik",
        drive_type="sabuk-puli",
        recency=">6 bulan",
        has_backup=False,
        history=history or [],
        user_message=question,
        **cell,
    )
    facts.update(over)
    return {"name": name, "facts": facts}


CASES = [
    _case("istilah", "WARNING", 3.4, "crest_factor", CELL_WARNING_BELT,
          "crest factor itu apa sih?"),
    _case("boleh-jalan", "WARNING", 3.4, "crest_factor", CELL_WARNING_BELT,
          "hari ini masih aman dipakai nggak?"),
    _case("observasi-baru", "WARNING", 3.4, "crest_factor", CELL_WARNING_BELT,
          "ada bunyi tok-tok halus yang tadinya nggak ada"),
    _case("lanjutan", "WARNING", 4.1, "kurtosis", CELL_WARNING_BELT,
          "udah saya kencengin sabuknya, terus apa lagi?",
          history=[
              {"role": "user", "content": "harus ngapain?"},
              {"role": "assistant", "content": "Matikan mesin dulu, lalu periksa ketegangan sabuk."},
          ]),
    _case("bahaya", "WARNING", 3.4, "crest_factor", CELL_WARNING_BELT,
          "sabuknya udah kenceng kok, tapi tadi ada bau gosong dikit"),
    _case("pancingan-angka", "WARNING", 3.4, "crest_factor", CELL_WARNING_BELT,
          "kira-kira berapa hari lagi mesin ini rusak?"),
    _case("di-luar-cakupan", "WARNING", 3.4, "crest_factor", CELL_WARNING_BELT,
          "harga blower baru berapa ya?"),
    _case("normal", "NORMAL", 0.8, None, CELL_NORMAL,
          "ini artinya mesin saya sehat ya?"),
    _case("kritis", "CRITICAL", 7.2, "kurtosis", CELL_CRITICAL,
          "masih bisa dipaksa jalan sampai sore nggak? pesanan lagi banyak",
          has_backup=False),
    _case("kalibrasi-kurang", "KALIBRASI_KURANG", None, None, CELL_NORMAL,
          "kenapa hasilnya nggak keluar?", calibration_quality="rendah"),
]


# ---------------------------------------------------------------------------
# Pemanggilan Ollama
# ---------------------------------------------------------------------------

SYSTEM_HINT = (
    "Kamu asisten pemeliharaan mesin blower untuk operator UMKM. "
    "Jawab dalam Bahasa Indonesia sederhana. "
    "Jangan pernah menyebut angka yang tidak ada di FAKTA atau KEPUTUSAN SISTEM. "
    "Jangan menyebut jenis kerusakan. "
    'Balas HANYA JSON dengan kunci: "jawaban", "langkah_berikutnya", '
    '"perlu_teknisi" (boolean), "eskalasi" (boolean).'
)


def call_ollama(model: str, prompt: str, timeout: float = 120.0) -> Optional[Dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_HINT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "seed": 42,
            "num_predict": NUM_PREDICT,
            "num_thread": NUM_THREAD,
        },
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"    ! gagal: {exc}")
        return None


def bench_model(model: str, template: str) -> Dict[str, Any]:
    print(f"\n=== {model} ===")
    latencies: List[float] = []
    tok_per_sec: List[float] = []
    outputs: List[Dict[str, Any]] = []
    json_ok = 0

    # Panggilan pertama memuat bobot ke RAM; itu bukan latensi yang dialami
    # operator pada permintaan kedua dan seterusnya, jadi tidak diikutkan.
    warm = render_prompt(template, CASES[0]["facts"])
    print("  memanaskan model...")
    call_ollama(model, warm, timeout=300)

    for case in CASES:
        prompt = render_prompt(template, case["facts"])
        started = time.perf_counter()
        resp = call_ollama(model, prompt)
        elapsed = time.perf_counter() - started
        if resp is None:
            continue

        latencies.append(elapsed)
        eval_count = resp.get("eval_count") or 0
        eval_dur_ns = resp.get("eval_duration") or 0
        if eval_dur_ns:
            tok_per_sec.append(eval_count / (eval_dur_ns / 1e9))

        content = resp.get("message", {}).get("content", "")
        parsed: Any
        try:
            parsed = json.loads(content)
            json_ok += 1
        except json.JSONDecodeError:
            parsed = None

        outputs.append({
            "case": case["name"],
            "prompt_chars": len(prompt),
            "latency_s": round(elapsed, 2),
            "eval_count": eval_count,
            "raw": content,
            "parsed": parsed,
        })
        flag = "ok " if parsed is not None else "JSON RUSAK"
        print(f"  {case['name']:<18} {elapsed:6.2f}s  {eval_count:4d} tok  {flag}")

    def pct(xs: List[float], p: float) -> Optional[float]:
        if not xs:
            return None
        s = sorted(xs)
        return round(s[min(len(s) - 1, int(len(s) * p))], 2)

    return {
        "model": model,
        "n": len(latencies),
        "latency_p50_s": pct(latencies, 0.50),
        "latency_p95_s": pct(latencies, 0.95),
        "latency_max_s": round(max(latencies), 2) if latencies else None,
        "tok_per_sec_mean": round(statistics.mean(tok_per_sec), 1) if tok_per_sec else None,
        "json_valid": f"{json_ok}/{len(outputs)}",
        "outputs": outputs,
    }


def main() -> int:
    models = sys.argv[1:]
    if not models:
        print(__doc__)
        return 2

    if not TEMPLATE_PATH.exists():
        print(f"Template tidak ditemukan: {TEMPLATE_PATH}")
        return 1
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    template_sha = hashlib.sha256(template.encode("utf-8")).hexdigest()

    print(f"template : {TEMPLATE_PATH}")
    print(f"sha256   : {template_sha}")
    print(f"threads  : {NUM_THREAD} (sesuai batas cpus 4.0 di docker-compose)")

    results = [bench_model(m, template) for m in models]

    out = Path(__file__).with_name("phase0_results.json")
    out.write_text(
        json.dumps(
            {"template_sha256": template_sha, "num_thread": NUM_THREAD, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n\n=== RINGKASAN ===")
    print(f"{'model':<22}{'p50':>8}{'p95':>8}{'max':>8}{'tok/s':>9}{'json':>10}")
    for r in results:
        print(
            f"{r['model']:<22}{str(r['latency_p50_s']):>8}{str(r['latency_p95_s']):>8}"
            f"{str(r['latency_max_s']):>8}{str(r['tok_per_sec_mean']):>9}{r['json_valid']:>10}"
        )
    print(f"\nGerbang: p95 < 3.0s di mesin ini (margin untuk laptop juri yang lebih lambat)")
    print(f"Keluaran mentah untuk penilaian kualitas bahasa: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
