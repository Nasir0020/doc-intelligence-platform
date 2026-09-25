"""
generator.py
============
The final stage of the RAG pipeline: takes the reranked candidate
chunks (Module 6's output) and a user's question, prompts an LLM to
answer USING ONLY those chunks, and parses the response into a
structured answer with verified citations.

Why "using only those chunks" is the central design goal here:
    Without explicit grounding instructions, an LLM will happily answer
    from its own general training knowledge, which defeats the entire
    purpose of a RAG system built specifically so answers are traceable
    to a specific uploaded document. Every design choice in this file
    (the system prompt wording, the citation format, parsing citations
    back out of the response) exists to make answers VERIFIABLE against
    the source documents, not just plausible-sounding.

Which other files use this:
    api/routes/query.py (Module 8) calls generate_answer() as the final
    step after retrieval + reranking.
"""

import logging
import re

from openai import OpenAI


from app.db.vector_store import SearchResult
from app.retrieval.schemas import Citation, GeneratedAnswer

logger = logging.getLogger(__name__)

GENERATION_MODEL = "llama3:latest"

SYSTEM_PROMPT = """You are a document Q&A assistant. You answer questions using ONLY the \
numbered source excerpts provided below — never your own general knowledge.

Rules:
1. Every factual claim in your answer must be immediately followed by a citation \
marker like [Source 1] referencing which excerpt it came from. If a claim draws on \
multiple sources, cite all of them, e.g. [Source 1][Source 3].
2. If the provided sources do NOT contain enough information to answer the question, \
say so explicitly and do not guess or use outside knowledge. Begin your response with \
"INSUFFICIENT_CONTEXT:" in that case, followed by a brief explanation.
3. Be concise. Do not repeat the source excerpts verbatim at length — synthesize and \
cite them.
"""

_CITATION_PATTERN = re.compile(r"\[Source (\d+)\]")


def _build_user_prompt(question: str, sources: list[SearchResult]) -> str:
    """
    Builds the numbered source block the model sees, alongside the
    question. Each source is labeled [Source N] — this exact numbering
    is what the model is instructed to cite, and what we parse back out
    of its response afterward, so the numbering scheme here and the
    regex pattern above must stay in sync.
    """
    source_blocks = []
    for i, source in enumerate(sources, start=1):
        pages = ", ".join(str(p) for p in source.page_numbers)
        source_blocks.append(f"[Source {i}] (page(s): {pages})\n{source.text}")

    joined_sources = "\n\n".join(source_blocks)
    return f"SOURCES:\n\n{joined_sources}\n\nQUESTION: {question}"


def _extract_citations(
    answer_text: str, sources: list[SearchResult]
) -> list[Citation]:
    """
    Finds every [Source N] marker actually used in the model's response
    and maps it back to the corresponding chunk's real metadata.

    Why re-derive citations from the response text instead of just
    returning ALL the sources we sent the model?
        The model may not have actually used every source we provided
        (reranking gives us the top-K most likely candidates, not a
        guarantee every one is relevant to the final answer). Returning
        only the citations the model actually referenced keeps the
        citation list HONEST — it reflects what was actually used to
        construct this specific answer, not just what was available.
    """
    cited_numbers = {int(n) for n in _CITATION_PATTERN.findall(answer_text)}

    citations = []
    for source_number in sorted(cited_numbers):
        index = source_number - 1
        if 0 <= index < len(sources):
            source = sources[index]
            citations.append(
                Citation(
                    source_number=source_number,
                    chunk_id=source.chunk_id,
                    source_filename=source.chunk_id.rsplit("_chunk_", 1)[0],
                    page_numbers=source.page_numbers,
                )
            )
        else:
            logger.warning(
                "Model cited [Source %d] which is out of range (only %d sources "
                "were provided) — dropping this citation as invalid.",
                source_number,
                len(sources),
            )

    return citations


def generate_answer(question: str, sources: list[SearchResult]) -> GeneratedAnswer:
    """
    The public entrypoint: calls the LLM with grounded sources and
    returns a structured, citation-verified answer.
    """
    if not sources:
        return GeneratedAnswer(
            answer_text=(
                "No relevant sources were found in the indexed documents for "
                "this question."
            ),
            citations=[],
            grounded=False,
        )

    client = OpenAI(
        base_url="http://host.docker.internal:11434/v1",
        api_key="ollama",
    )

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_prompt(question, sources),
            },
        ],
        max_tokens=1024,
    )

    answer_text = response.choices[0].message.content or ""

    grounded = not answer_text.startswith("INSUFFICIENT_CONTEXT:")
    citations = _extract_citations(answer_text, sources) if grounded else []

    return GeneratedAnswer(
        answer_text=answer_text,
        citations=citations,
        grounded=grounded,
    )