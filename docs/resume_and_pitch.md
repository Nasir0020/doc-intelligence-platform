# Resume, Portfolio & Pitch Content

## Resume bullet points (ATS-friendly)

- Built a production-style multi-modal RAG platform (Python, FastAPI, Next.js/TypeScript) enabling natural-language Q&A over scanned and native PDFs with page-level source citations
- Designed a hybrid retrieval pipeline combining BM25 keyword search and dense vector embeddings via Reciprocal Rank Fusion, improving retrieval recall over vector-only search
- Implemented a two-stage retrieval architecture (bi-encoder recall + cross-encoder reranking) balancing query latency against answer precision
- Built an OCR-aware document ingestion pipeline (PyMuPDF, Tesseract, pdfplumber) with automatic detection of scanned vs. native-text pages and layout-aware, token-budgeted chunking
- Engineered a citation-verification system that cross-checks every LLM-generated citation against retrieved sources, dropping unsupported/hallucinated references
- Deployed a full MLOps stack: MLflow experiment tracking for retrieval-quality metrics (Recall@K, Precision@K, MRR), Dockerized microservices (Postgres, Weaviate, FastAPI, Next.js), and GitHub Actions CI running tests against a live Postgres service container
- Achieved 88% test coverage across 32 automated tests (pytest), including integration tests through the live HTTP API and defensive-coding regression tests for real bugs found during development
- Evaluated and made a pragmatic build-vs-adopt decision against the RAGAS evaluation framework after identifying a hard dependency conflict, implementing a lightweight custom evaluation harness instead

## Technical skills (for a resume skills section)

**Languages**: Python, TypeScript
**AI/ML**: RAG architecture, hybrid retrieval (BM25 + dense embeddings), cross-encoder reranking, embedding models (sentence-transformers), prompt engineering, LLM grounding/citation verification, evaluation metrics (Recall@K, Precision@K, MRR)
**Backend**: FastAPI, Pydantic, SQLAlchemy, PostgreSQL, Weaviate (vector DB)
**Frontend**: Next.js, React, Tailwind CSS
**MLOps/Infra**: Docker, Docker Compose, MLflow, GitHub Actions (CI/CD), Railway, Vercel
**Testing**: pytest, integration testing, dependency mocking
**Document processing**: PyMuPDF, Tesseract OCR, pdfplumber

## GitHub repository description

> A production-style multi-modal document intelligence platform (RAG). Upload scanned or native PDFs, ask natural-language questions, get answers with page-level citations. Hybrid retrieval (BM25 + vector), cross-encoder reranking, FastAPI + Next.js, full MLOps stack (MLflow, Docker Compose, CI/CD with a live Postgres service container). 88% test coverage.

## LinkedIn project post

> Just shipped a project I'm genuinely proud of: a document intelligence platform that lets you upload messy real-world PDFs — scanned contracts, tables, reports — and ask questions in plain English, with every answer citing the exact page it came from.
>
> What made this more than a weekend RAG demo: hybrid retrieval combining keyword and semantic search (pure vector search alone misses exact terms like invoice numbers), a cross-encoder reranking stage for precision, and a citation-verification step that catches and drops any source the model hallucinates rather than trusting it blindly.
>
> On the engineering side: FastAPI + Postgres + Weaviate backend, Next.js/TypeScript frontend, MLflow for tracking retrieval-quality experiments, Docker Compose orchestration, and a CI pipeline that runs tests against a real Postgres service container. 88% test coverage, 32 automated tests.
>
> I also made a call I'm glad I made: I evaluated the RAGAS framework for evaluation, hit a real dependency conflict, and built a lighter custom metrics harness instead rather than forcing a fragile fix — sometimes the right engineering decision is knowing when *not* to use the "standard" tool.
>
> Repo: [link] | Happy to talk through any part of the architecture.

## Portfolio site description (short form)

> **Document Intelligence Platform** — A multi-modal RAG system for querying scanned and native PDFs with cited, grounded answers. Hybrid BM25+vector retrieval, cross-encoder reranking, FastAPI/Next.js, full MLOps pipeline. [Live demo] · [GitHub] · [Architecture doc]

## HR-round introduction (~30 seconds, conversational)

"I built a document intelligence platform — it's a RAG system, so it lets you upload PDFs and ask questions in plain English, with answers that cite the exact source page. I focused on making it genuinely production-style rather than a demo: proper testing, CI/CD, MLOps tracking, real error handling. It's the project I'd point to as best representing how I actually work — methodically, testing as I go, and being upfront when something doesn't work rather than papering over it."

## Technical-round introduction (~90 seconds — see the longer walkthrough already prepared)

Use the "Longer version (interview walkthrough)" content from earlier in this project's history — it covers the ingestion pipeline, hybrid retrieval, reranking, grounded generation, and the engineering stack, ending on an open invitation for the interviewer to steer into whichever part they want to go deeper on.
