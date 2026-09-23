# Multi-Modal Document Intelligence Platform

A retrieval-augmented generation (RAG) system for querying mixed-content
documents (scanned PDFs, tables, reports) in natural language, with
citations back to the source page/section.

> **Status:** Module 1 (project scaffolding) complete. See `docs/` for
> architecture details as each module is built.

## Local Setup

### Prerequisites
- Python 3.12+
- Docker & Docker Compose (for later modules — Postgres, Weaviate, MLflow)

### 1. Clone and create a virtual environment
```bash
git clone <your-repo-url>
cd doc-intelligence-platform
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
cp .env.example .env
# then edit .env with real values (e.g. your Anthropic API key)
```

### 4. Run the backend locally (without Docker)
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```
Visit `http://localhost:8000/docs` for interactive API docs, or
`http://localhost:8000/health/` for a liveness check.

### 5. Run with Docker Compose (once Postgres/Weaviate are added in later modules)
```bash
docker compose up --build
```

## Project Structure

See `docs/architecture.md` (added in Module 8) for the full system
architecture diagram and rationale.

## Deployment

**Recommended path (PaaS):** Railway for the backend + Postgres + Weaviate, and Vercel for the Next.js frontend.

1. Push this repository to GitHub.
2. On Railway, create a project from the GitHub repository. Railway uses `railway.json` and `backend/Dockerfile` for the backend deployment. Add a Postgres service and configure `ANTHROPIC_API_KEY` and other required environment variables.
3. On Vercel, import the repository and set the root directory to `frontend/`. Configure `NEXT_PUBLIC_API_BASE_URL` to point to the deployed Railway backend.
4. For custom domains, DNS configuration and TLS details, see `docs/deployment_guide.md`.

**Self-hosted alternative:** `nginx.conf` and `docker-compose.yml` provide the configuration for a VPS/EC2 deployment with a reverse proxy and TLS.

**CI/CD:** `.github/workflows/ci.yml` runs the backend test suite against a real PostgreSQL service container and builds the frontend on pushes and pull requests. Railway and Vercel can automatically deploy from the connected GitHub repository.

## Roadmap

- [x] Module 1: Project scaffolding & configuration
- [x] Module 2: Document parsing & OCR ingestion
- [x] Module 3: Layout-aware chunking
- [x] Module 4: Embeddings & vector indexing
- [x] Module 5: Hybrid retrieval
- [x] Module 6: Cross-encoder reranking
- [x] Module 7: LLM generation & citation
- [x] Module 8: Full FastAPI backend
- [x] Module 9: Evaluation harness
- [x] Module 10: Frontend (Next.js)
- [x] Module 11: MLOps (MLflow, persistence & Docker Compose)
- [x] Module 12: Testing (32 tests, 88% coverage)
- [x] Module 13: Cloud deployment
- [x] Module 14: Documentation & interview prep
