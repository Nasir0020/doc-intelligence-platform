"""
chunker.py
==========
Converts a ParsedDocument (from document_processor.py) into a list of
Chunks — token-budgeted units of content ready for embedding.

Why this file is needed:
    Embedding models and vector search work on fixed-size units of text
    (chunks), not whole documents. HOW we draw the boundaries between
    chunks directly affects retrieval quality — this is one of the
    highest-leverage design decisions in the entire RAG pipeline, and
    is a near-guaranteed interview topic ("how did you decide on your
    chunking strategy?").

Which other files use this:
    embedder.py (Module 4) will call chunk_document() and embed each
    resulting Chunk.
"""

import re

from app.ingestion.schemas import Chunk, ChunkType, ParsedDocument, ParsedPage
from app.ingestion.token_counter import count_tokens

MAX_CHUNK_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64

# Splits text into paragraphs on blank lines (one or more consecutive
# newlines with only whitespace between them). This works well for
# NATIVE-extracted text, which usually preserves the document's real
# paragraph breaks. It intentionally does NOT try to split OCR text
# this way — explained in _page_to_blocks below.
_PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n")


class _Block:
    """
    An internal (non-Pydantic, plain Python) intermediate representation
    of one "unit" of content before it's been grouped into a final Chunk.

    Why a plain class here instead of another Pydantic model?
        This is a purely internal implementation detail of the chunking
        algorithm — it never crosses a module boundary or gets
        serialized to JSON/API responses. Pydantic's validation
        overhead buys us nothing here; a lightweight plain class (or
        we could have used a dataclass — see the note below) is more
        appropriate for a short-lived internal helper.
    """

    def __init__(self, text: str, page_number: int, block_type: ChunkType):
        self.text = text
        self.page_number = page_number
        self.block_type = block_type
        self.token_count = count_tokens(text)


def _page_to_blocks(page: ParsedPage) -> list[_Block]:
    """
    Breaks one page's content into ordered blocks: paragraphs (for
    native text) or the whole OCR'd page as one block, plus one block
    per table.

    Why OCR text isn't paragraph-split like native text:
        Tesseract's plain output (as we built it in ocr_parser.py)
        doesn't reliably preserve paragraph breaks — it returns
        recognized words joined by spaces, without a strong signal of
        where one paragraph ends and another begins. Splitting it on
        blank lines that likely don't exist would just produce one giant
        "paragraph" anyway. Rather than pretend we have paragraph
        structure we don't, we treat the whole OCR page as a single
        block and let the chunk-size limit (not paragraph boundaries)
        decide where it gets split if it's too long. This is a
        documented, deliberate simplification — a more advanced version
        could use Tesseract's layout-analysis mode to recover
        approximate paragraph geometry, which we note as a future
        improvement rather than silently pretending this limitation
        doesn't exist.
    """
    blocks: list[_Block] = []

    if page.text.strip():
        if page.extraction_method.value == "native":
            paragraphs = [
                p.strip() for p in _PARAGRAPH_SPLIT_PATTERN.split(page.text) if p.strip()
            ]
        else:
            paragraphs = [page.text.strip()]

        for paragraph in paragraphs:
            blocks.append(_Block(paragraph, page.page_number, ChunkType.TEXT))

    for table in page.tables:
        blocks.append(
            _Block(table.to_markdown(), page.page_number, ChunkType.TABLE)
        )

    return blocks


# Splits on sentence boundaries: a period/question mark/exclamation
# mark followed by whitespace. `(?<=[.!?])` is a LOOKBEHIND assertion —
# it checks that the preceding character was sentence-ending punctuation
# WITHOUT consuming/removing it from the split pieces (unlike a normal
# split, which would discard the matched separator). `\s+` then matches
# and consumes the actual whitespace between sentences.
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _split_oversized_block(block: "_Block") -> list["_Block"]:
    """
    A single paragraph can, on its own, exceed MAX_CHUNK_TOKENS (this was
    caught by real testing on a long-form document — not a hypothetical
    edge case). When that happens, we fall back to splitting that one
    paragraph at sentence boundaries and re-grouping sentences up to the
    token budget, rather than emitting one oversized chunk that would
    dilute embedding quality just like naive fixed-size chunking would.

    If a single SENTENCE somehow still exceeds the budget on its own
    (rare, but possible with unusually long run-on sentences or OCR
    text with no punctuation at all), we accept it as a single
    over-budget chunk and note this as a known, documented edge case
    rather than pretending it can't happen.
    """
    if block.token_count <= MAX_CHUNK_TOKENS:
        return [block]

    sentences = [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(block.text) if s.strip()]

    sub_blocks: list[_Block] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if current_tokens + sentence_tokens > MAX_CHUNK_TOKENS and current:
            sub_blocks.append(
                _Block(" ".join(current), block.page_number, block.block_type)
            )
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        sub_blocks.append(_Block(" ".join(current), block.page_number, block.block_type))

    return sub_blocks


def _blocks_to_chunks(blocks: list[_Block], source_filename: str) -> list[Chunk]:
    """
    Groups an ordered list of blocks into token-budgeted Chunks, with
    overlap carried between consecutive TEXT chunks.

    The algorithm, in plain language:
        Walk through blocks in order, accumulating them into a
        "current chunk" until adding the next block would exceed
        MAX_CHUNK_TOKENS. At that point, close off the current chunk,
        start a new one, and seed it with the tail end of the chunk we
        just closed (the overlap) so context isn't lost at the seam.
        Table blocks are ALWAYS their own standalone chunk — never
        merged with surrounding text — regardless of how small they are
        relative to the token budget.
    """
    chunks: list[Chunk] = []
    current_texts: list[str] = []
    current_pages: set[int] = set()
    current_tokens = 0
    chunk_index = 0

    def _flush():
        """Closes off the current in-progress chunk, if it has content."""
        nonlocal chunk_index
        if not current_texts:
            return
        text = "\n\n".join(current_texts)
        chunks.append(
            Chunk(
                chunk_id=f"{source_filename}_chunk_{chunk_index:04d}",
                source_filename=source_filename,
                chunk_type=ChunkType.TEXT,
                text=text,
                token_count=count_tokens(text),
                page_numbers=sorted(current_pages),
            )
        )
        chunk_index += 1

    for block in blocks:
        if block.block_type == ChunkType.TABLE:
            # Close whatever text chunk was in progress, emit the table
            # as its own standalone chunk, then continue with a fresh
            # text accumulator afterward.
            _flush()
            current_texts, current_pages, current_tokens = [], set(), 0

            chunks.append(
                Chunk(
                    chunk_id=f"{source_filename}_chunk_{chunk_index:04d}",
                    source_filename=source_filename,
                    chunk_type=ChunkType.TABLE,
                    text=block.text,
                    token_count=block.token_count,
                    page_numbers=[block.page_number],
                )
            )
            chunk_index += 1
            continue

        # If adding this block would blow the budget, close the current
        # chunk first, then seed the next one with an overlap tail.
        if current_tokens + block.token_count > MAX_CHUNK_TOKENS and current_texts:
            _flush()

            # Build the overlap: walk backward through the text we just
            # closed off, accumulating whole paragraphs (never a partial
            # paragraph) until we've collected roughly CHUNK_OVERLAP_TOKENS
            # worth, or run out of paragraphs.
            #
            # BUG FOUND VIA TESTING: if the only available paragraph is
            # itself much larger than CHUNK_OVERLAP_TOKENS (common with
            # long single-paragraph pages), naively including "at least
            # one whole paragraph" pulls in the ENTIRE oversized
            # paragraph as overlap, nearly doubling the next chunk. Fix:
            # when a paragraph alone would blow the remaining overlap
            # budget, take only a truncated TAIL SLICE of it sized to
            # approximately fit the remaining budget, rather than the
            # whole paragraph.
            overlap_texts: list[str] = []
            overlap_tokens = 0
            for prev_text in reversed(current_texts):
                prev_tokens = count_tokens(prev_text)
                remaining_budget = CHUNK_OVERLAP_TOKENS - overlap_tokens

                if prev_tokens <= remaining_budget:
                    overlap_texts.insert(0, prev_text)
                    overlap_tokens += prev_tokens
                    continue

                if remaining_budget > 0:
                    # Approximate how many trailing characters correspond
                    # to `remaining_budget` tokens, using this paragraph's
                    # own char-to-token ratio as the estimate. This is a
                    # heuristic slice (may cut mid-word) — acceptable
                    # here because overlap only needs to preserve nearby
                    # CONTEXT, not be a clean standalone unit.
                    approx_chars = max(
                        1, int(len(prev_text) * (remaining_budget / prev_tokens))
                    )
                    tail_fragment = prev_text[-approx_chars:]

                    # The character-to-token ratio above is only an estimate.
                    # Trim the fragment until the actual tokenizer confirms
                    # that the overlap stays within the remaining token budget.
                    while count_tokens(tail_fragment) > remaining_budget and len(tail_fragment) > 1:
                        tail_fragment = tail_fragment[1:]

                    overlap_texts.insert(0, tail_fragment)
                    overlap_tokens += count_tokens(tail_fragment)

                break  # either way, don't look further back

            # The overlap calculation is approximate, so validate the
            # actual serialized chunk against the real tokenizer before
            # adding the next block. Trim overlap from the front until the
            # candidate stays within MAX_CHUNK_TOKENS.
            while overlap_texts and count_tokens(
                "\n\n".join(overlap_texts + [block.text])
            ) > MAX_CHUNK_TOKENS:
                overlap_texts.pop(0)

            current_texts = overlap_texts
            current_pages = {block.page_number}  # reset, re-added below
            current_tokens = count_tokens("\n\n".join(current_texts))

        current_texts.append(block.text)
        current_pages.add(block.page_number)
        current_tokens = count_tokens("\n\n".join(current_texts))

    _flush()  # emit whatever's left after the loop ends
    return chunks


def chunk_document(document: ParsedDocument) -> list[Chunk]:
    """
    The public entrypoint for this module: turns a fully parsed document
    into its final list of embeddable Chunks.
    """
    all_blocks: list[_Block] = []
    for page in document.pages:
        for block in _page_to_blocks(page):
            if block.block_type == ChunkType.TEXT:
                # Text blocks get expanded if they're individually too
                # large. Tables are deliberately NOT run through this —
                # we keep tables atomic even if oversized, documented as
                # a known limitation, since splitting a table mid-row
                # would corrupt its structure far worse than leaving it
                # slightly over budget.
                all_blocks.extend(_split_oversized_block(block))
            else:
                all_blocks.append(block)

    return _blocks_to_chunks(all_blocks, document.source_filename)
