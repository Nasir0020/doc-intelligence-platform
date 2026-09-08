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

# Copying requirements.txt BEFORE the rest of the code is a deliberate
# Docker layer-caching optimization: Docker caches each instruction as a
# layer. If only application code changes (not dependencies), Docker
# reuses the cached "pip install" layer instead of re-running it,
# making rebuilds dramatically faster during development.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
