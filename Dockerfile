# ==============================================================================
# AUDIAX AI SERVICE — PRODUCTION MULTI-STAGE DOCKERFILE
# Architecture: Senior MLOps Standards (Anthropic/Google Cloud Production Grade)
# Base: Python 3.11 Slim (Debian Bookworm)
# ==============================================================================

# ------------------------------------------------------------------------------
# STAGE 1: Dependency Builder
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY ai/requirements.txt /build/ai-requirements.txt
COPY service/requirements.txt /build/service-requirements.txt

# Install PyTorch CPU + Torchaudio + AI & Service dependencies
# Using PyTorch CPU wheel keeps the image size lightweight (~1.2GB vs ~8GB CUDA)
# which is optimal for audio anomaly detection inference on standard CPU instances.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /build/ai-requirements.txt \
    -r /build/service-requirements.txt

# ------------------------------------------------------------------------------
# STAGE 2: Minimal Runtime Environment
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="Audiax MLOps Team" \
      description="Audiax Audio Anomaly Detection AI Microservice" \
      version="1.0.0"

# Install runtime OS dependencies:
# - libsndfile1: Required by soundfile for low-latency wav processing
# - ffmpeg: Audio codec fallback & preprocessing
# - curl: Container healthcheck probing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-privileged user and group for security compliance
RUN groupadd -g 10001 audiax && \
    useradd -u 10001 -g audiax -s /bin/bash -m audiax

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application modules
COPY --chown=audiax:audiax ai/ /app/ai/
COPY --chown=audiax:audiax service/ /app/service/

RUN mkdir -p /app/models /app/ai/weights /app/ai/vendor/beats /app/tmp && \
    chown -R audiax:audiax /app

# ------------------------------------------------------------------------------
# Verifikasi aset — gagal saat BUILD, bukan saat runtime.
#
# `COPY ai/ /app/ai/` di atas ikut membawa checkpoint dan vendor BEATs, tapi
# hanya kalau file itu memang ada di mesin yang mem-build. Kalau tidak ada,
# COPY tetap sukses dan image jadi lolos build dalam keadaan rusak: kegagalan
# baru muncul saat runtime sebagai /healthz 503 tanpa penjelasan, dan itu
# ditemukan di saat paling buruk -- di depan juri.
#
# Lebih baik build-nya yang gagal, sekarang, dengan instruksi yang bisa dibaca.
# ------------------------------------------------------------------------------
RUN set -e; \
    if [ ! -f /app/ai/vendor/beats/BEATs.py ]; then \
        echo "" >&2; \
        echo "BUILD GAGAL: source BEATs tidak ada di ai/vendor/beats/." >&2; \
        echo "Jalankan di host lalu build ulang:  python scripts/download_assets.py" >&2; \
        echo "" >&2; \
        exit 1; \
    fi; \
    if [ ! -f /app/ai/weights/beats_finetuned.pt ] \
       && [ ! -f /app/ai/weights/BEATs_iter3_plus_AS20K.pt ]; then \
        echo "" >&2; \
        echo "BUILD GAGAL: checkpoint BEATs tidak ada di ai/weights/." >&2; \
        echo "Checkpoint sengaja tidak masuk git (361 MB, di atas batas GitHub)." >&2; \
        echo "Jalankan di host lalu build ulang:  python scripts/download_assets.py" >&2; \
        echo "" >&2; \
        exit 1; \
    fi; \
    echo "Aset terverifikasi:"; \
    ls -la /app/ai/weights/ /app/ai/vendor/beats/

# Switch to non-root user
USER audiax

# Environment configurations
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    AUDIAX_BEATS_SOURCE=/app/ai/vendor/beats \
    AUDIAX_BEATS_CHECKPOINT=/app/ai/weights/beats_finetuned.pt

# Expose HTTP API Port
EXPOSE 8000

# Docker Healthcheck: Monitors /healthz which verifies model warm-up in memory
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Production Entrypoint: Single worker recommended for PyTorch in-process concurrency
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--access-log"]
