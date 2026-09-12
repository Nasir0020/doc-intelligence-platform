"""
token_counter.py
================
Provides a single function, count_tokens(), used everywhere we need to
know how "big" a piece of text is in TOKENS rather than characters.

Why tokens, not characters or words?
    Every embedding model and LLM has a context window measured in
    TOKENS, not characters or words. A "token" is a sub-word unit
    produced by the model's tokenizer (e.g., "unbelievable" might split
    into "un", "believ", "able" — three tokens). Character count is a
    poor proxy: "iiiii" (5 chars) is 1-2 tokens, while "extraordinarily"
    (15 chars) might be 3-4 tokens. If we sized chunks by character
    count alone, we'd systematically over- or under-fill the model's
    actual token budget depending on the text's vocabulary.

Why this file has a FALLBACK path:
    tiktoken (OpenAI's tokenizer library) downloads its BPE merge-rule
    file from OpenAI's servers on first use, and caches it locally after
    that. In network-restricted environments (locked-down corporate
    networks, air-gapped deployments, or this development sandbox — we
    hit this exact failure while building this module), that first
    download can fail. A production system should NOT crash the entire
    ingestion pipeline just because an exact tokenizer isn't reachable —
    it should degrade to a reasonable approximation and log a clear
    warning, which is exactly what we do here.

Which other files use this:
    chunker.py calls count_tokens() constantly while deciding where to
    place chunk boundaries.
"""

import logging

logger = logging.getLogger(__name__)

# A widely-cited, well-tested rule of thumb for English text: roughly
# 4 characters per token on average (OpenAI's own documentation states
# this same approximation). It's not exact — it will be off for
# non-English text, code, or unusual vocabulary — but it degrades
# gracefully rather than failing outright.
_APPROX_CHARS_PER_TOKEN = 4

_encoding = None
_tried_loading_tiktoken = False


def _get_tiktoken_encoding():
    """
    Lazily attempts to load tiktoken's encoding, exactly once per
    process. Returns None if it's unavailable, in which case callers
    fall back to the character-based approximation.

    Why lazy (only load on first actual use, not at import time)?
        Importing this module shouldn't trigger a network call. If a
        caller never actually needs token counting in a given run, we
        shouldn't pay the cost (or risk the failure) of trying to fetch
        tiktoken's data file at all.

    Why a module-level global instead of recomputing every call?
        Loading the encoding is the expensive/network-dependent part.
        Once we know it succeeded (or failed), every subsequent call
        should reuse that result instantly rather than repeating the
        expensive attempt.
    """
    global _encoding, _tried_loading_tiktoken

    if _tried_loading_tiktoken:
        return _encoding

    _tried_loading_tiktoken = True
    try:
        import tiktoken

        _encoding = tiktoken.get_encoding("cl100k_base")
        logger.info("tiktoken encoding loaded successfully — using exact token counts.")
    except Exception as exc:
        # Deliberately broad `except Exception`, not a specific
        # exception type: tiktoken can fail for several different
        # reasons in the wild (network error, permissions error on its
        # cache directory, corrupted download) and we want ALL of them
        # to fall back gracefully rather than crashing ingestion.
        logger.warning(
            "tiktoken unavailable (%s: %s) — falling back to an approximate "
            "character-based token estimate (~%d chars/token). Token counts "
            "will be approximate, not exact.",
            type(exc).__name__,
            exc,
            _APPROX_CHARS_PER_TOKEN,
        )
        _encoding = None

    return _encoding


def count_tokens(text: str) -> int:
    """
    Returns the token count for a piece of text — exact if tiktoken is
    available, approximate otherwise.
    """
    encoding = _get_tiktoken_encoding()

    if encoding is not None:
        return len(encoding.encode(text))

    # Approximate fallback: character count divided by our documented
    # average. `max(1, ...)` ensures even a tiny non-empty string counts
    # as at least 1 token, avoiding a zero-token chunk being treated as
    # "free" by the chunking logic below.
    return max(1, len(text) // _APPROX_CHARS_PER_TOKEN)
