# Interview Question Bank

Organized by category, weighted toward what this project actually demonstrates. Each question has three answer depths, a common mistake to avoid, a likely follow-up, and what a recruiter/interviewer is actually listening for.

---

## RAG & Retrieval (core to this project)

### 1. Why not just use vector search alone for retrieval?
- **Beginner**: Vector search finds semantically similar text but can miss exact terms.
- **Intermediate**: Dense embeddings compress meaning into a fixed-size vector; rare/specific tokens (IDs, exact numbers) can get diluted relative to the overall semantic signal, so exact-match queries underperform.
- **Industry**: This is why hybrid retrieval (BM25 + vector, fused via RRF or a weighted scheme) is standard in production RAG — BM25's exact-term matching and vector search's semantic matching are complementary failure modes, not redundant.
- **Common mistake**: Saying "vector search is just worse" — it's not worse, it fails on a *different* class of queries.
- **Follow-up**: "How would you fuse two rankings with incompatible score scales?" -> Reciprocal Rank Fusion, working with ranks not raw scores.
- **Recruiter is listening for**: whether you understand *why*, not just that hybrid search exists as a buzzword.

### 2. Walk me through Reciprocal Rank Fusion.
- **Beginner**: It combines two ranked lists into one by rewarding items that rank highly in either list.
- **Intermediate**: `RRF_score(d) = sum of 1/(k + rank_i(d))` across each retrieval method; k is typically 60. Items appearing in both lists, or ranked highly in either, accumulate higher fused scores.
- **Industry**: RRF sidesteps the score-scale-mismatch problem — BM25 and cosine similarity aren't on comparable numeric scales, but ranks always are. It's also robust to one method returning an item the other completely misses (that item just gets 0 contribution from the missing method, not penalized further).
- **Common mistake**: Trying to normalize and average raw scores instead — technically possible but fragile and corpus-dependent.
- **Follow-up**: "Why k=60 specifically?" -> It's the value from the original Cormack et al. paper; became a de facto default, not a magic number.

### 3. Why do you need a reranking stage if you already have retrieval?
- **Beginner**: Reranking is more accurate but too slow to run on everything, so you narrow down first.
- **Intermediate**: Bi-encoders embed query and document independently (fast, precomputable); cross-encoders process them jointly (slow, but let attention directly compare query and document tokens — much higher precision).
- **Industry**: Two-stage retrieval is the standard production pattern: cheap recall over the full corpus, then expensive precision over a small candidate pool. This is directly analogous to search engines' "candidate generation to ranking" architecture.
- **Common mistake**: Thinking reranking replaces retrieval — it operates only on retrieval's output.
- **Follow-up**: "What happens if your candidate pool is too small?" -> The true best answer might not even be in the pool for reranking to find — recall from stage 1 upper-bounds precision at stage 2.

### 4. How do you prevent an LLM from hallucinating in a RAG system?
- **Beginner**: Tell it to only use the provided context.
- **Intermediate**: Explicit grounding instructions in the system prompt, an escape hatch for "insufficient context," and citation requirements.
- **Industry**: Instructions alone aren't sufficient — verify citations *after* generation by parsing them out of the response and cross-checking against what was actually retrieved, dropping any citation pointing outside the valid range (models do hallucinate citation numbers). This makes the system defensible even when the model doesn't perfectly follow instructions.
- **Common mistake**: Trusting the model's citations without verification.
- **Follow-up**: "How would you measure hallucination rate at scale?" -> This is what RAGAS's faithfulness metric does (decompose the answer into claims, verify each against context) — or a custom claim-verification pass.

### 5. How would you choose a chunk size for a RAG pipeline?
- **Beginner**: Not too small, not too big.
- **Intermediate**: Too small loses context per chunk; too large dilutes the embedding's semantic focus (mixing topics) and wastes context budget on irrelevant content.
- **Industry**: There's no universally correct number — it's an empirical tradeoff you validate with retrieval metrics (Recall@K, Precision@K) on a golden dataset, not guessed. Document structure matters more than a fixed token count: respecting paragraph/table boundaries beats naive fixed-size splitting regardless of the exact size chosen.
- **Common mistake**: Citing a specific number (e.g. "512 tokens") as if it's universally correct rather than something you'd tune and measure.
- **Follow-up**: "How does overlap help, and what's the tradeoff?" -> Preserves context at chunk boundaries; tradeoff is redundant storage/embedding cost.

### 6. What's the difference between Recall@K, Precision@K, and MRR?
- **Beginner**: Recall = did you find it; precision = how much of what you found was relevant; MRR = how early you found it.
- **Intermediate**: Recall@K is binary per query (found at least one relevant result in top K or not), Precision@K is the fraction of top K that's relevant, MRR = mean of 1/rank of the first relevant result across queries.
- **Industry**: These measure different failure modes — high recall/low precision means retrieval finds the right stuff but buries it in noise; low recall means retrieval fundamentally can't find the answer, no amount of reranking or better prompting fixes that downstream. MRR specifically matters when the *first* result quality matters most.
- **Follow-up**: "Which would you prioritize optimizing first, and why?" -> Recall — nothing downstream can fix a retrieval miss.

---

## Embeddings & NLP

### 7. Explain how cosine similarity works and why it's used over Euclidean distance for embeddings.
- **Beginner**: It measures the angle between two vectors, not the distance between their positions.
- **Intermediate**: `cos_sim = (A.B)/(|A||B|)` — normalizes out vector magnitude, so it compares *direction* only.
- **Industry**: Embedding magnitude often correlates with factors unrelated to meaning (e.g., text length), so direction-only comparison is more robust for semantic similarity. Many systems pre-normalize embeddings to unit length, making cosine similarity and dot product equivalent — enabling faster dot-product-optimized vector search.
- **Common mistake**: Not knowing that normalized vectors make cosine similarity and dot product mathematically identical.
- **Follow-up**: "Why would a vector DB offer dot product as a separate metric if it's equivalent?" -> Dot product is computationally cheaper (skips the normalization division at query time) when vectors are pre-normalized at index time.

### 8. What's a cross-encoder vs. a bi-encoder?
(Same concept as Retrieval Q3, asked from a different angle.)
- **Recruiter is listening for**: whether your understanding is conceptual (works from any phrasing) or memorized (only works with one exact wording).

### 9. What is BM25 and how does it differ from TF-IDF?
- **Beginner**: BM25 is a smarter version of TF-IDF for search ranking.
- **Intermediate**: Both weight terms by frequency and rarity (IDF), but BM25 adds term-frequency *saturation* (diminishing returns for repeated terms via k1) and document-length normalization (via b) — raw TF-IDF grows linearly with term count, which BM25 deliberately avoids.
- **Industry**: BM25 remains the standard for keyword/lexical search in production systems (Elasticsearch's default scoring, Weaviate's hybrid search) precisely because of these refinements — it's a genuinely complementary signal even in a neural-search-first world, not obsolete technology.
- **Follow-up**: "What does increasing k1 do?" -> Makes the score more sensitive to term frequency (less saturation).

---

## Python & Engineering Fundamentals

### 10. What's the difference between `async def` and `def` in a FastAPI route?
- **Beginner**: `async def` is for asynchronous code.
- **Intermediate**: `async def` routes run on the event loop and should only contain non-blocking (`await`-using) operations; `def` routes are automatically run in a thread pool by FastAPI, which is actually safer for CPU-bound or blocking synchronous code.
- **Industry**: A blocking call inside an `async def` route stalls the *entire event loop*, freezing every other concurrent request on that worker — a subtle, easy-to-introduce production bug. Heavy CPU work (like ML inference) is often better pushed to a background task queue regardless of route style, so the web worker stays responsive.
- **Common mistake**: Assuming `async def` is always "faster" — only for I/O-bound work; CPU-bound work doesn't benefit and can actively harm the app if it blocks the loop.
- **Follow-up**: "How would you handle a slow OCR call in an async route?" -> Run it in a thread pool explicitly, or push to a background worker and return a job ID immediately.

### 11. Explain Python's `@lru_cache` and when it becomes a bug.
- **Beginner**: It caches function results so repeated calls are faster.
- **Intermediate**: A memoization decorator keyed on arguments; useful for expensive, deterministic operations like loading a model once.
- **Industry**: In request-serving code, it's how you get true singleton behavior across requests. But in a test suite, cached state silently leaks between tests unless explicitly cleared, causing order-dependent failures — genuinely hard to diagnose because tests pass or fail depending on execution order.
- **Follow-up**: "How do you reset it for tests?" -> `function.cache_clear()`, typically in an `autouse` pytest fixture.
- **Recruiter is listening for**: real experience with this exact gotcha, not a textbook definition.

### 12. What's a context manager and why use `with`?
- **Beginner**: A pattern that automatically cleans up resources.
- **Intermediate**: Guarantees a cleanup action (closing a file, a DB session) runs even if an exception is raised inside the block.
- **Industry**: The standard idiom for anything with acquire/release semantics. Skipping it is a classic source of resource leaks in production, especially under error conditions that are easy to forget to test.
- **Follow-up**: "How does a FastAPI dependency using `yield` relate to this?" -> Same pattern — code before `yield` is setup, code after is guaranteed cleanup, applied per-request.

---

## Databases & SQL

### 13. Why use `unique=True` at the database level instead of checking in application code first?
- **Beginner**: The database enforces it more reliably.
- **Intermediate**: Application-level "check then insert" has a race condition window — two concurrent requests can both pass the check before either inserts.
- **Industry**: A classic TOCTOU (time-of-check to time-of-use) bug class. A database-level unique constraint is atomic; the application only needs to catch the resulting integrity error and translate it into a clean response (e.g., 409 Conflict).
- **Follow-up**: "What status code fits this, and why not 400?" -> 409 Conflict — the request is well-formed but conflicts with existing state, a materially different situation from malformed input.

### 14. What does `pool_pre_ping=True` do in SQLAlchemy, and why does it matter?
- **Beginner**: It checks the connection is alive before using it.
- **Intermediate**: Connection pools can hand out a stale connection (DB restarted, a load balancer dropped an idle TCP connection) — without pre-ping, the first real query on it fails with a confusing error.
- **Industry**: Trades a small latency cost per checkout for eliminating an entire class of intermittent, hard-to-reproduce production errors.

---

## System Design (project-scaled)

### 15. How would you scale this RAG system to handle 10x the document volume?
- **Beginner**: Add more servers.
- **Intermediate**: Move ingestion (parsing, OCR, embedding) to a background job queue so uploads don't block on slow processing; Weaviate and Postgres both scale independently of the API layer.
- **Industry**: Identify the actual bottleneck first — OCR/embedding are CPU/GPU-bound and parallelize well; vector query latency degrades with corpus size unless using approximate-nearest-neighbor indexing; the LLM call is usually the fixed latency floor regardless of corpus size. Horizontal scaling helps throughput, not a single request's end-to-end latency, which is dominated by the sequential retrieve-rerank-generate chain.
- **Follow-up**: "Where's the bottleneck at 10x specifically?" -> Almost certainly reranking (cost scales with candidate pool) and the LLM call, not retrieval.
- **Recruiter is listening for**: reasoning about specific bottlenecks, not a generic "add more servers" non-answer.

### 16. Why separate ingestion and retrieval into different subsystems?
- **Beginner**: They do different things.
- **Intermediate**: Different performance profiles — ingestion is batch/I-O-heavy and latency-tolerant; retrieval must be synchronous and low-latency.
- **Industry**: This separation enables independently scaling and testing each — adding ingestion worker capacity during a bulk upload doesn't touch query-serving capacity, and a bug in OCR parsing can't take down the live query endpoint. Direct application of separation-of-concerns to genuinely different non-functional requirements, not just code aesthetics.

---

## MLOps & Testing

### 17. Why is a hand-rolled evaluation harness sometimes the right call instead of an established library like RAGAS?
- **Beginner**: Sometimes the library doesn't work in your environment.
- **Intermediate**: Established libraries carry their own dependency footprint; a genuine, unresolvable dependency conflict is a legitimate reason to build something narrower and fully controlled.
- **Industry**: The judgment isn't "avoid third-party tools" — it's recognizing when forcing a fragile fix (downgrading unrelated packages) creates more risk than it resolves, and building a smaller tool covering your actual immediate need while documenting the "real" tool's correct integration path for later. Also cheaper for CI — LLM-judge metrics cost money/time on every run; a statistical harness doesn't.
- **Recruiter is listening for**: pragmatism and honesty about tradeoffs, not dogma in either direction.

### 18. What's the difference between a unit test and an integration test, concretely, in your project?
- **Beginner**: Unit tests test one function; integration tests test multiple things together.
- **Intermediate**: My chunker/retriever/generator unit tests construct objects directly and mock external calls; my API tests go through the real FastAPI `TestClient`, exercising real routing, dependency injection, and a real database.
- **Industry**: Integration tests catch bugs unit tests structurally can't — I found a real bug where the query endpoint called the embedder before checking if the corpus was empty, an edge case only manifesting when multiple real components interact in a specific order. Unit tests are faster and pinpoint failures precisely; integration tests are slower but catch component-boundary bugs.

---

*This is a starting set, weighted toward the depth an interviewer would actually probe given this specific project. Want me to keep expanding categories (Docker/Linux, transformer internals, more SQL, coding-round DS&A), or move into interactive mock interview rounds — one question at a time, you answer, I evaluate and give the ideal response — the way the original mentorship format specified?*
