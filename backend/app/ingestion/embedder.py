"""
embedder.py
===========
Converts text (document chunks, and later, user queries) into dense
vector embeddings.

Production backend:
    sentence-transformers loading BAAI/bge-base-en-v1.5 (768 dimensions).
    This is a strong, widely-used open-source embedding model that
    ranks well on the MTEB retrieval benchmark, and is free to
    self-host (no per-token API cost, unlike hosted embedding APIs).

Fallback backend (used automatically when the neural model can't be
downloaded — e.g. this sandbox has no route to huggingface.co):
    TF-IDF + Truncated SVD, a classical technique called Latent
    Semantic Analysis (LSA). It's a genuinely real, historically
    important embedding approach (used in production search systems
    long before neural embeddings existed) — not a fake placeholder.

IMPORTANT ARCHITECTURAL DIFFERENCE between the two backends, worth
understanding deeply because it's a real interview-worthy distinction:
    The neural backend is a STATELESS FUNCTION — it can embed any text,
    including text it's never seen, independently and identically every
    time, because "meaning" is baked into the model's pretrained weights.
    The fallback backend is STATEFUL — TF-IDF's vocabulary and SVD's
    projection axes are learned FROM a specific corpus of documents at
    fit time. It cannot meaningfully embed a brand-new document (or a
    user's query) unless that fitted vocabulary/projection is reused.
    This is exactly why classical LSA-style approaches fell out of favor
    for production RAG: every new batch of documents technically
    changes the "meaning space," while a pretrained neural model's
    space is fixed and consistent across arbitrary new text forever.
    We keep this real limitation visible rather than hiding it.

Which other files use this:
    document ingestion (Module 8 will wire this into the upload
    endpoint) calls embed_documents() once per batch of chunks.
    retrieval/retriever.py (Module 5) calls embed_query() per user query.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

NEURAL_MODEL_NAME = "BAAI/bge-base-en-v1.5"
NEURAL_MODEL_DIM = 768

# Cap on the fallback's vector dimensionality. TruncatedSVD requires the
# number of output components to be strictly less than min(n_samples,
# n_features), so this is an upper bound, not a guarantee — see
# _fit_fallback below for how we adapt it to small corpora.
FALLBACK_MAX_DIM = 256


class Embedder:
    """
    A single object that owns whichever backend is available, and
    exposes the same two methods (embed_documents / embed_query)
    regardless of which backend is active underneath — callers never
    need an if/else on which backend is in play.

    This is the Strategy design pattern: the "algorithm" (how text
    becomes a vector) is swappable behind a stable interface, decided
    once at construction time.
    """

    def __init__(self):
        self._neural_model = self._try_load_neural_model()
        self._fallback_vectorizer = None
        self._fallback_svd = None
        self.backend_name = "neural" if self._neural_model else "tfidf_svd_fallback"
        self.dimension: int | None = NEURAL_MODEL_DIM if self._neural_model else None

    @staticmethod
    def _try_load_neural_model():
        """
        Attempts to load the sentence-transformers model. Returns None
        (rather than raising) on any failure, exactly mirroring the
        graceful-degradation pattern used in token_counter.py — a
        module unable to reach an external dependency should degrade,
        not crash the whole ingestion pipeline.
        """
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(NEURAL_MODEL_NAME)
            logger.info("Loaded neural embedding model '%s'.", NEURAL_MODEL_NAME)
            return model
        except Exception as exc:
            logger.warning(
                "Neural embedding model unavailable (%s: %s) — falling back to "
                "a local TF-IDF + SVD embedding backend. Retrieval quality will "
                "be lexical/statistical rather than semantic until this is "
                "resolved in an environment with model-hub access.",
                type(exc).__name__,
                str(exc)[:150],
            )
            return None

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """
        Embeds a batch of documents (chunks). Returns an array of shape
        (len(texts), dimension).

        For the fallback backend specifically, this is also where
        fitting happens (see the class docstring's note on statefulness)
        — the first call to embed_documents establishes the fallback's
        vocabulary and projection axes for this corpus.
        """
        if self._neural_model is not None:
            # normalize_embeddings=True scales each vector to unit
            # length. Since cosine similarity already divides out
            # magnitude, this doesn't change the ranking — but many
            # vector databases (including Weaviate) offer a faster
            # "dot product" distance metric that's mathematically
            # equivalent to cosine similarity ONLY when vectors are
            # pre-normalized. Normalizing here lets us use that faster
            # metric downstream without changing results.
            return self._neural_model.encode(texts, normalize_embeddings=True)

        return self._fit_fallback(texts)

    def embed_query(self, text: str) -> np.ndarray:
        """Embeds a single user query using whichever backend is active."""
        if self._neural_model is not None:
            return self._neural_model.encode([text], normalize_embeddings=True)[0]

        if self._fallback_vectorizer is None or self._fallback_svd is None:
            raise RuntimeError(
                "The TF-IDF/SVD fallback backend must be fit on a document "
                "corpus (via embed_documents) BEFORE it can embed a query — "
                "unlike the neural backend, it has no meaning outside the "
                "corpus it was fit on. Call embed_documents() first."
            )

        tfidf_vector = self._fallback_vectorizer.transform([text])
        return self._fallback_svd.transform(tfidf_vector)[0]

    def _fit_fallback(self, texts: list[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._fallback_vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
        )
        tfidf_matrix = self._fallback_vectorizer.fit_transform(texts)

        # TruncatedSVD's n_components must be strictly less than
        # min(n_samples, n_features) — with a very small test corpus
        # (a handful of chunks), FALLBACK_MAX_DIM (256) could be larger
        # than that. We clamp it down rather than let sklearn raise an
        # opaque error.
        max_valid_dim = min(FALLBACK_MAX_DIM, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        n_components = max(1, max_valid_dim)

        self._fallback_svd = TruncatedSVD(n_components=n_components, random_state=42)
        vectors = self._fallback_svd.fit_transform(tfidf_matrix)
        self.dimension = n_components

        logger.info(
            "Fitted TF-IDF+SVD fallback embedder: %d documents -> %d-dimensional vectors.",
            len(texts),
            n_components,
        )
        return vectors


_embedder_instance: Embedder | None = None


def get_embedder() -> Embedder:
    """Lazy singleton, same rationale as config.get_settings() — avoid
    re-attempting the (potentially slow or network-dependent) model
    load on every call."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance
