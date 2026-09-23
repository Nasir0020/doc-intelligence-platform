# Deployment Guide

## Recommended path: Railway (backend) + Vercel (frontend)

### 1. Backend + Postgres + Weaviate on Railway
1. Push this repo to GitHub.
2. On [railway.app](https://railway.app): New Project → Deploy from GitHub repo → select this repo.
3. Railway auto-detects `railway.json` and builds `backend/Dockerfile`.
4. Add a Postgres plugin (one click in Railway's dashboard) — it automatically injects `DATABASE_URL`-style connection variables.
5. Add a Weaviate service (Railway's template marketplace has one, or deploy the official Weaviate Docker image directly).
6. In the backend service's Variables tab, set: `ANTHROPIC_API_KEY`, `POSTGRES_*` (or point at Railway's auto-injected Postgres variables), `WEAVIATE_URL`.

### 2. Frontend on Vercel
1. On [vercel.com](https://vercel.com): Import Project → select this repo → set root directory to `frontend/`.
2. Vercel auto-detects Next.js (see `frontend/vercel.json`).
3. Set `NEXT_PUBLIC_API_BASE_URL` to your Railway backend's public URL.

### 3. Custom domain (optional)
- Vercel: add a `CNAME` record pointing to `cname.vercel-dns.com`.
- Railway: add a `CNAME` for a subdomain (e.g. `api.yourdomain.dev`) pointing to Railway's provided target.
- Both platforms auto-issue Let's Encrypt TLS certificates once DNS propagates — no manual Certbot step.

## Alternative: self-hosted (VPS / EC2)

Use `docker-compose.yml` directly on any machine with Docker installed:
```bash
cp .env.example .env   # fill in real values
docker compose up --build -d
```
Then put `nginx.conf` in front of it for TLS termination and routing (see the file's own comments for the full Certbot setup: `certbot --nginx -d yourdomain.com`).

**Honest status**: neither the Docker Compose multi-service stack nor `nginx.conf` were runtime-verified end-to-end in the sandbox this project was built in — Docker itself wasn't available there. Both are written to each tool's standard, documented configuration and individually-verified patterns (the backend Dockerfile ran standalone; Postgres/SQLAlchemy logic was verified via SQLite), but the full multi-container orchestration should be your first thing to test on your own machine.

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`:
- Backend: full pytest suite against a **real Postgres service container** (not a substitute), plus a direct round-trip verification against it
- Frontend: `npm audit` (fails the build on any high/critical vulnerability) + production build

Railway and Vercel both auto-deploy on push to `main` once connected via their own GitHub integrations — no separate deploy job needed in this workflow.

## Environment variables reference

See `.env.example` (backend) and `frontend/.env.local.example` (frontend) for the full list. In production, set these in each platform's dashboard — never commit real values to `.env`.
