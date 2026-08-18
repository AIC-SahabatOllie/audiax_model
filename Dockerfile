# syntax=docker/dockerfile:1
#
# Multi-stage build. Konsekuensi: menjalankan (docker compose up) TIDAK
# memerlukan koneksi internet sama sekali -- mitigasi risiko kegagalan demo
# akibat jaringan panitia. Membangun (docker compose build) BUTUH internet
# sekali untuk memasang dependensi Python.
#
# PRASYARAT sebelum build (lihat README.md "Setup"):
#   ai/vendor/beats/{BEATs.py,backbone.py,modules.py}  <- source resmi Microsoft
#   ai/weights/{beats_finetuned.pt,adapter.pt}          <- hasil repo pelatihan
#
# Catatan dependensi: image ini SENGAJA memasang dari ai/requirements.txt +
# service/requirements.txt (bukan requirements.txt di root, yang isinya
# konsolidasi SEMUA dependensi -- termasuk pandas/sklearn/jupyter untuk dev
# lokal -- dan tidak dimaksudkan untuk image produksi yang ramping).
#
# Image ini JALAN SENDIRI sebagai layanan HTTP mandiri (bukan sekadar base
# layer) -- konsekuensi dari keputusan 3 deployment terpisah (FE/BE/AI).
# service/ (lapisan HTTP tipis) dipasang di atas ai/ (logika inti murni),
# tapi dependensinya sengaja dipasang dari file terpisah (lihat dua baris
# COPY requirements di bawah) supaya pemisahan "ai/ nol dependensi web"
# tetap terlihat bahkan di level dependency install, bukan cuma di kode.

FROM python:3.11-slim AS builder

WORKDIR /build
COPY ai/requirements.txt ./requirements.txt
COPY service/requirements.txt ./service-requirements.txt
RUN pip install --no-cache-dir --target=/deps -r requirements.txt -r service-requirements.txt

FROM python:3.11-slim AS runtime

WORKDIR /app
COPY --from=builder /deps /usr/local/lib/python3.11/site-packages
COPY ai/ ./ai/
COPY service/ ./service/

# Verifikasi saat build: kalau vendor/weights lupa disiapkan, gagal SEKARANG
# (saat build), bukan nanti diam-diam saat container sudah jalan.
RUN test -f ai/vendor/beats/BEATs.py || (echo "FATAL: ai/vendor/beats/BEATs.py tidak ada -- lihat README.md Setup" && exit 1)
RUN test -f ai/weights/beats_finetuned.pt || (echo "FATAL: ai/weights/beats_finetuned.pt tidak ada -- lihat README.md Setup" && exit 1)

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=4)" || exit 1

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
