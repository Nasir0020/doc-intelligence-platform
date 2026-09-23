# syntax=docker/dockerfile:1

# --- Base image ---
# We pin to a specific Python version + slim variant deliberately:
# "slim" images are based on Debian but strip out extras (docs, compilers)
# not needed at runtime, keeping the final image small. We avoid "alpine"
# variants for Python specifically because many ML/data libraries (numpy,
# pandas, etc. we'll add in later modules) ship pre-built wheels for
# glibc-based distros (like Debian) but not for musl-based Alpine, which
# would force slow from-source compilation during every image build.
FROM python:3.12-slim

# Prevents Python from writing .pyc files and buffers stdout, which
# matters for seeing logs in real time when running `docker logs`.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# --- System dependencies for Module 2 (OCR) ---
# tesseract-ocr is a system-level binary, NOT a Python package — pip
# cannot install it. This is a real, common gotcha: a requirements.txt
# passing locally on a machine that happens to already have tesseract
# installed (like our dev sandbox) can still fail in a fresh container
# that doesn't have it. Docker images must explicitly install every
# system dependency the app needs, not just the Python ones.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copying requirements.txt BEFORE the rest of the code is a deliberate
# Docker layer-caching optimization: Docker caches each instruction as a
# layer. If only application code changes (not dependencies), Docker
# reuses the cached "pip install" layer instead of re-running it,
# making rebuilds dramatically faster during development.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app

# --- Running as a non-root user ---
# Run the application as an unprivileged user instead of root.
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Docker-native health check for deployment platforms.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
