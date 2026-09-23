# Architecture

## System overview

```mermaid
graph TB
    subgraph Client
        FE[Next.js Frontend]
    end

    subgraph Backend["FastAPI Backend"]
        UP["/documents/upload"]
        Q["/query"]
        DEP[Dependencies: retriever, reranker singletons]
    end

    subgraph Ingestion["Ingestion Pipeline"]
        PARSE[PDF Parser<br/>native text + OCR]
        CHUNK[Chunker<br/>token-budgeted, boundary-aware]
        EMB[Embedder<br/>neural or TF-IDF+SVD fallback]
    end

    subgraph Retrieval["Retrieval Pipeline"]
        HYBRID[Hybrid Retriever<br/>BM25 + vector, RRF fusion]
        RERANK[Cross-Encoder Reranker]
        GEN[Generator<br/>Claude API, grounded + cited]
    end

    subgraph Storage
        PG[(Postgres<br/>document metadata)]
        VEC[(Weaviate / InMemory<br/>vector index)]
    end

    subgraph MLOps
        MLFLOW[MLflow<br/>retrieval-quality tracking]
    end

    FE -->|HTTP| UP
    FE -->|HTTP| Q

    UP --> PARSE --> CHUNK --> EMB
    EMB --> VEC
    UP --> PG

    Q --> DEP --> HYBRID
    HYBRID --> VEC
    HYBRID --> RERANK --> GEN
    GEN -->|Anthropic API| FE

    EMB -.->|evaluation harness| MLFLOW
```

## Why this shape

Ingestion and retrieval are deliberately separate subsystems (see Module 1's rationale) — they have different performance characteristics: ingestion is batch-oriented and I/O-heavy (PDF parsing, OCR), while retrieval must be low-latency and synchronous. Keeping them as independent pipelines means they can be reasoned about, tested, and eventually scaled independently.

## The two request flows

### Upload flow

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant API as FastAPI
    participant Parse as Parser/OCR
    participant Chunk as Chunker
    participant Emb as Embedder
    participant DB as Postgres
    participant Vec as Vector Store

    U->>FE: Selects a PDF
    FE->>API: POST /documents/upload
    API->>API: Validate file type, check duplicate filename
    API->>Parse: process_pdf()
    Parse->>Parse: Native text extraction per page
    Parse->>Parse: OCR fallback for low-text pages
    Parse-->>API: ParsedDocument
    API->>Chunk: chunk_document()
    Chunk-->>API: list[Chunk]
    API->>Emb: embed_documents()
    Emb-->>API: vectors
    API->>Vec: index chunks + vectors
    API->>DB: persist document metadata
    API-->>FE: DocumentSummary (201)
    FE-->>U: Shows indexed document
```

### Query flow

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant API as FastAPI
    participant Emb as Embedder
    participant Ret as Hybrid Retriever
    participant Rank as Reranker
    participant LLM as Claude API

    U->>FE: Asks a question
    FE->>API: POST /query
    API->>API: Check corpus not empty
    API->>Emb: embed_query()
    Emb-->>API: query vector
    API->>Ret: search() — BM25 + vector, RRF-fused
    Ret-->>API: candidate chunks
    API->>Rank: rerank()
    Rank-->>API: top-k reranked chunks
    API->>LLM: generate_answer() with grounded prompt
    LLM-->>API: answer text with [Source N] citations
    API->>API: Extract + verify citations
    API-->>FE: GeneratedAnswer (answer + citations)
    FE-->>U: Displays answer with page-level citations
```

## Key design decisions and their rationale

| Decision | Why |
|---|---|
| Hybrid retrieval (BM25 + vector) over vector-only | Pure semantic search under-ranks exact terms (invoice numbers, regional codes) |
| Two-stage retrieval (fast recall → precise rerank) | Cross-encoders are too slow to run over an entire corpus; bi-encoders can't match a cross-encoder's precision |
| Citations re-derived from the response text, not echoed from input | Keeps citations honest — reflects what the model actually used, not everything it was given |
| Every network-dependent component has a tested local fallback | This project was built in a network-restricted sandbox; the same pattern (tokenizer, embedder, reranker) also protects a real deployment against a model-hub outage |
| Ingestion/retrieval separated into independent subsystems | Different performance profiles (batch/I-O-heavy vs. low-latency/synchronous); independently testable and scalable |
