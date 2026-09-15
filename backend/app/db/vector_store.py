"""
vector_store.py
================
Defines the interface for storing and searching chunk embeddings, plus
two implementations:

  1. WeaviateVectorStore — the PRODUCTION backend. Requires a running
     Weaviate server (via Docker Compose, added in Module 11). We could
     NOT verify this against a real server in this sandbox (no Docker
     available here, as noted back in Module 1) — written to Weaviate's
     documented v4 client API, but flagged as untested-in-this-environment
     rather than falsely claimed to be verified.

  2. InMemoryVectorStore — a DEV/TEST-ONLY backend: brute-force cosine
     similarity search using NumPy, no external server required. This
     is what let us actually run and verify retrieval logic in this
     sandbox. It does not scale (it's O(n) per query, holds everything
     in process memory, and vanishes on restart) — it exists purely so
     we can test the RETRIEVAL ALGORITHM independently of whether a
     real vector database is reachable, exactly like the fallback
     patterns in token_counter.py and embedder.py.

Why an abstract base class (interface) instead of just writing
WeaviateVectorStore directly:
    retriever.py (Module 5) should be able to work against EITHER
    backend without changing a single line — this is the Dependency
    Inversion principle again (also used in embedder.py's Strategy
    pattern): code that consumes a vector store should depend on the
    abstract CAPABILITY ("add vectors", "search vectors"), not on
    Weaviate specifically.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from app.config import settings


@dataclass
class SearchResult:
    """
    A single vector search hit.

    Why @dataclass here instead of a plain class (like _Block in
    chunker.py) or a Pydantic BaseModel (like Chunk in schemas.py)?
        @dataclass is a middle ground: like a plain class, it has no
        runtime validation overhead (we don't need it here — these
        objects are constructed internally by our own trusted code, not
        parsed from external/untrusted input like an API request body).
        Unlike a plain class, @dataclass auto-generates __init__,
        __repr__ (a readable print representation), and __eq__ (so two
        SearchResults with equal fields compare as equal) FOR us, from
        just the field declarations below — less boilerplate than
        writing __init__ by hand as we did for _Block.
    """

    chunk_id: str
    score: float
    text: str
    page_numbers: list[int]


class VectorStore(ABC):
    """
    Abstract base class defining the contract every vector store
    backend must fulfill.

    `ABC` (Abstract Base Class) and the `@abstractmethod` decorator
    below work together: Python will refuse to let you even
    INSTANTIATE a subclass of VectorStore unless it has provided a
    concrete implementation of every method marked @abstractmethod.
    This is enforced at runtime, not just a style convention — it's
    Python's mechanism for guaranteeing every backend genuinely
    implements the full interface.
    """

    @abstractmethod
    def add(
        self,
        chunk_ids: list[str],
        vectors: np.ndarray,
        texts: list[str],
        page_numbers: list[list[int]],
    ) -> None:
        """Indexes a batch of chunks with their embeddings."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        """Returns the top_k most similar chunks to the query vector."""
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    """
    DEV/TEST-ONLY backend. Holds all vectors in a single in-process
    NumPy array and computes cosine similarity against every stored
    vector on every search call (brute force — no approximate-nearest-
    neighbor indexing, unlike a real vector DB).
    """

    def __init__(self):
        self._vectors: np.ndarray | None = None
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []
        self._page_numbers: list[list[int]] = []

    def add(
        self,
        chunk_ids: list[str],
        vectors: np.ndarray,
        texts: list[str],
        page_numbers: list[list[int]],
    ) -> None:
        if self._vectors is None:
            self._vectors = vectors.copy()
        else:
            # np.vstack stacks arrays row-wise — appending new chunk
            # vectors as additional rows onto the existing matrix.
            self._vectors = np.vstack([self._vectors, vectors])

        self._chunk_ids.extend(chunk_ids)
        self._texts.extend(texts)
        self._page_numbers.extend(page_numbers)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        if self._vectors is None or len(self._chunk_ids) == 0:
            return []

        # Vectorized cosine similarity against EVERY stored vector at
        # once, rather than looping in Python. `self._vectors @ query_vector`
        # is matrix-vector multiplication: each row (one chunk's vector)
        # gets dotted with the query vector in a single NumPy operation.
        # This is dramatically faster than a Python for-loop over rows
        # for anything beyond a handful of vectors, because NumPy
        # pushes the actual arithmetic down into compiled C code.
        dot_products = np.sum(self._vectors * query_vector, axis=1)
        norms = np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(query_vector)
        # Avoid division by zero for any degenerate all-zero vector.
        norms = np.where(norms == 0, 1e-10, norms)
        similarities = dot_products / norms

        # argsort gives indices that WOULD sort the array ascending;
        # [::-1] reverses that to descending (most similar first);
        # [:top_k] takes only the first top_k of those indices.
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            SearchResult(
                chunk_id=self._chunk_ids[i],
                score=float(similarities[i]),
                text=self._texts[i],
                page_numbers=self._page_numbers[i],
            )
            for i in top_indices
        ]


class WeaviateVectorStore(VectorStore):
    """
    PRODUCTION backend. NOT verified against a live server in this
    sandbox (no Docker/Weaviate available here) — implemented against
    the documented weaviate-client v4 API and will be verified once
    Module 11 stands up the real Docker Compose stack.

    Why Weaviate specifically (recapping the Module 1 decision): it
    supports HYBRID search (combining vector similarity with keyword/
    BM25 search) natively, which matters for documents containing exact
    identifiers (invoice numbers, dates) that pure semantic search can
    under-rank relative to keyword search.
    """

    COLLECTION_NAME = "DocumentChunk"

    def __init__(self):
        import weaviate

        # connect_to_local() targets a Weaviate instance running on
        # localhost (as our Docker Compose setup will provide) — this
        # call itself will raise a connection error in THIS sandbox
        # since no such server is running here.
        self._client = weaviate.connect_to_local(
            host=settings.WEAVIATE_URL.replace("http://", "").split(":")[0]
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        import weaviate.classes.config as wc

        if not self._client.collections.exists(self.COLLECTION_NAME):
            self._client.collections.create(
                name=self.COLLECTION_NAME,
                properties=[
                    wc.Property(name="chunk_id", data_type=wc.DataType.TEXT),
                    wc.Property(name="text", data_type=wc.DataType.TEXT),
                    wc.Property(
                        name="page_numbers", data_type=wc.DataType.INT_ARRAY
                    ),
                ],
                # Explicitly disabling Weaviate's own built-in
                # vectorizer modules: WE compute embeddings ourselves
                # (via embedder.py) so every chunk goes through the
                # SAME model regardless of which backend Weaviate might
                # otherwise auto-select, keeping embedding generation
                # centralized and swappable in one place (embedder.py),
                # not split across two systems.
                vectorizer_config=wc.Configure.Vectorizer.none(),
            )

    def add(
        self,
        chunk_ids: list[str],
        vectors: np.ndarray,
        texts: list[str],
        page_numbers: list[list[int]],
    ) -> None:
        collection = self._client.collections.get(self.COLLECTION_NAME)
        with collection.batch.dynamic() as batch:
            for chunk_id, vector, text, pages in zip(
                chunk_ids, vectors, texts, page_numbers
            ):
                batch.add_object(
                    properties={
                        "chunk_id": chunk_id,
                        "text": text,
                        "page_numbers": pages,
                    },
                    vector=vector.tolist(),
                )

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        collection = self._client.collections.get(self.COLLECTION_NAME)
        response = collection.query.near_vector(
            near_vector=query_vector.tolist(),
            limit=top_k,
            return_metadata=["distance"],
        )

        results = []
        for obj in response.objects:
            # Weaviate returns COSINE DISTANCE (0 = identical), not
            # similarity — we convert back to the same similarity scale
            # (1 = identical) our InMemoryVectorStore uses, so callers
            # get consistent semantics regardless of backend.
            distance = obj.metadata.distance or 0.0
            results.append(
                SearchResult(
                    chunk_id=obj.properties["chunk_id"],
                    score=1.0 - distance,
                    text=obj.properties["text"],
                    page_numbers=obj.properties["page_numbers"],
                )
            )
        return results
