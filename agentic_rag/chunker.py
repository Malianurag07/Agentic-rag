"""Chunking (plan section 5.3), done per subsection so a chunk never crosses a heading.

    paragraphs   packed up to ~doc_chunk_words; an over-long paragraph is cut on sentence boundaries
    rules        a "MANDATORY CROSS-DOMAIN RULE [R-02 ...]" paragraph is always its own chunk, tagged with its rule id
    tables       rows rendered as "Column: value | Column: value" and packed up to ~table_chunk_words; the table title and
                 column names are repeated on every chunk so a row is never separated from its meaning
    figures      kept as flagged placeholders (needs_vision) because their content is not in the text"""

import hashlib
import re
from dataclasses import dataclass, field

from .blocks import Block
from .config import Config
from .structure import Subsection
from .text import sentences

RULE_ID = re.compile(r"\bR-0?(\d)\b")


@dataclass
class DocChunk:
    kind: str                       # paragraph | rule | table | figure
    text: str
    page_start: int | None = None
    page_end: int | None = None
    needs_vision: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        norm = re.sub(r"[^a-z0-9]+", " ", self.text.lower()).strip()
        return hashlib.sha1(norm.encode()).hexdigest()[:16]


def _pack(pieces: list[tuple[str, int | None]], limit: int) -> list[tuple[str, int | None, int | None]]:
    out, buf, words, p0, p1 = [], [], 0, None, None
    for text, page in pieces:
        n = len(text.split())
        if buf and words + n > limit:
            out.append((" ".join(buf), p0, p1))
            buf, words, p0, p1 = [], 0, None, None
        buf.append(text)
        words += n
        p0 = page if p0 is None else p0
        p1 = page if page is not None else p1
    if buf:
        out.append((" ".join(buf), p0, p1))
    return out


def _pack_rows(rows: list[str], limit: int) -> list[list[str]]:
    groups: list[list[str]] = []
    buf: list[str] = []
    words = 0
    for row in rows:
        n = len(row.split())
        if buf and words + n > limit:
            groups.append(buf)
            buf, words = [], 0
        buf.append(row)
        words += n
    if buf:
        groups.append(buf)
    return groups


FIGURE_CAPTION = re.compile(r"^(\[CHART/FIGURE SUMMARY\]\s*)?Figure\s+\d", re.IGNORECASE)


def _split_long(text: str, limit: int, overlap: int = 0) -> list[str]:
    if len(text.split()) <= limit:
        return [text]
    pieces = [t for t, _, _ in _pack([(s, None) for s in sentences(text)], limit)]
    if overlap <= 0:
        return pieces
    out = [pieces[0]]
    for prev, piece in zip(pieces, pieces[1:]):
        tail = " ".join(sentences(prev)[-overlap:])
        out.append(f"{tail} {piece}")
    return out


def table_row_text(header: list[str], row: list[str]) -> str:
    return " | ".join(f"{h}: {c}" if h else c for h, c in zip(header, row) if c)


def chunk_subsection(sub: Subsection, config: Config) -> list[DocChunk]:
    chunks: list[DocChunk] = []
    pending: list[tuple[str, int | None]] = []

    def flush() -> None:
        for text, p0, p1 in _pack(pending, config.doc_chunk_words):
            chunks.append(DocChunk("paragraph", text, p0, p1))
        pending.clear()

    for i, block in enumerate(sub.blocks):
        if block.kind == "para":
            if "MANDATORY CROSS-DOMAIN RULE" in block.text:
                flush()
                ids = sorted({f"R-0{d}" for d in RULE_ID.findall(block.text)})
                chunks.append(DocChunk("rule", block.text, block.page, block.page, meta={"rule_ids": ids}))
            else:
                for piece in _split_long(block.text, config.doc_chunk_words, config.chunk_overlap_sentences):
                    pending.append((piece, block.page))
        elif block.kind == "table":
            flush()
            rows = [table_row_text(block.header, r) for r in block.rows]
            last = block.meta.get("last_page", block.page)
            head = f"Table in {sub.number} {sub.title}. Columns: " + " | ".join(h for h in block.header if h) + "."
            groups = [[r] for r in rows] if config.table_row_chunks else _pack_rows(rows, config.table_chunk_words)
            for group in groups:
                chunks.append(DocChunk("table", head + " Rows:\n" + "\n".join(f"- {r}" for r in group), block.page, last,
                                       meta={"columns": block.header, "table_rows": len(block.rows)}))
        elif block.kind == "figure":
            flush()
            neighbours = [sub.blocks[j] for j in (i - 1, i + 1) if 0 <= j < len(sub.blocks)]
            caption = next((b.text for b in neighbours if b.kind == "para" and FIGURE_CAPTION.match(b.text)), "")
            text = (caption + " " if caption else "") + (block.text or "[Figure: an image with no extractable text]")
            chunks.append(DocChunk("figure", text, block.page, block.page, needs_vision=True,
                                   meta={**block.meta, "caption": caption}))
    flush()
    return chunks
