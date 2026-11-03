# Part 3: Finding the structure, then cutting into chunks

After loading, a document is a flat list of Blocks. Two steps turn that into something searchable. **Structure** discovers the table of contents hidden in the text (Section 3, then 3.1, 3.2 ...), which gives the pipeline its tree of topics. **Chunking** then cuts each subsection into pieces of a sensible size. Cutting happens *inside* a subsection, never across a heading, so a chunk never mixes two topics.

Analogy: loading is photocopying the book; structure is rebuilding the table of contents; chunking is cutting each chapter into index cards that each stand on their own.

## `agentic_rag/structure.py`

Builds the tree Section > Subsection > Blocks by recognising numbered headings in the paragraph text. Works the same for text, Word and PDF input.

### File description

**Lines 1-7**

```python
"""Turn a flat list of Blocks into the plan's tree: Section (topic) > numbered subsection (type) > blocks (text).

Recognises, in any of the three formats:
    "SECTION 3: TITLE" / "Section 3: Title"        -> a section
    "3.2 Title (Structured Matrix): body text..."  -> a subsection (title ends at the first colon; body may follow)
Anything before the first section is kept as section 0 (front matter); text between a section heading and its first
numbered subsection becomes an implicit "n.0 Introduction"."""
```

States the job and the two heading shapes it recognises: `SECTION 3: TITLE` and `3.2 Title: body...`. Text before the first section becomes 'section 0' (front matter). Text between a section heading and its first numbered subsection becomes an implicit 'n.0 Introduction'.

### Imports, patterns and limits

**Lines 9-10**

```python
import re
from dataclasses import dataclass, field
```

`re` for patterns; `dataclass` and `field` for records.

**Line 12**

```python
from .blocks import Block
```

Imports `Block`, the unit being organised.

**Line 14**

```python
SECTION_RE = re.compile(r"^\s*section\s+(\d{1,2})\s*[:.\-–]\s*(.+)$", re.IGNORECASE | re.DOTALL)
```

`SECTION_RE` matches a section heading: the word 'section' (any capitalisation), then a one or two digit number, then a separator (`:`, `.`, `-` or an en dash), then the title. `DOTALL` lets the title run across line breaks.

**Line 15**

```python
SUB_RE = re.compile(r"^\s*(\d{1,2})\.(\d{1,2})\s+(?=[A-Z\"“(])(.+)$", re.DOTALL)
```

`SUB_RE` matches a subsection heading: number.number (like `3.2`), spaces, then the title. The `(?=[A-Z"“(])` part is a *look-ahead*: the next character must be a capital letter, a quote or a bracket. This stops sentences like '3.5 days of leave' from being mistaken for headings.

**Line 16**

```python
MAX_TITLE = 190
```

`MAX_TITLE` caps a heading's length at 190 characters.

### The Subsection and Section records

**Lines 19-23**

```python
@dataclass
class Subsection:
    number: str                     # "3.2"
    title: str
    blocks: list = field(default_factory=list)
```

A Subsection has a number (like '3.2'), a title, and a list of the Blocks that belong to it.

**Lines 25-28**

```python
    def page_range(self) -> tuple[int | None, int | None]:
        pages = [b.page for b in self.blocks if b.page]
        last = [b.meta.get("last_page", b.page) for b in self.blocks if b.page]
        return (min(pages), max(last)) if pages else (None, None)
```

`page_range` reports the first and last PDF page of a subsection. It lists the pages of the Blocks that have one, and also uses `last_page` from tables that continued across pages. It returns (lowest, highest), or (None, None) when there are no pages (Word and text files).

**Lines 31-35**

```python
@dataclass
class Section:
    number: str                     # "3"
    title: str
    subsections: list = field(default_factory=list)
```

A Section has a number (like '3'), a title, and a list of its Subsections.

### Splitting a heading from its body

**Lines 38-41**

```python
def _split_title(text: str) -> tuple[str, str]:
    """A numbered paragraph is either a bare heading or "Heading: body". Return (title, body).
    A short paragraph with no sentence break is a heading in its own right, colon or not, so a title such as
    "UNSTRUCTURED POLICY PROSE: PERFORMANCE CALIBRATION, PIPS, AND MOONLIGHTING" is kept whole."""
```

`_split_title` takes the text after the number and returns two things: the title and the body. In many documents a subsection starts as 'Heading: body text...' in one paragraph, so the heading has to be separated.

**Lines 42-43**

```python
    if len(text) <= MAX_TITLE + 40 and not re.search(r"[.!?]\s", text) and not text.rstrip().endswith("."):
        return text.rstrip(": ").strip(), ""
```

A short paragraph (under about 230 characters) that has no sentence break and does not end in a full stop is a **heading on its own**. The whole text is the title, even if it contains a colon. This keeps titles like 'UNSTRUCTURED POLICY PROSE: PERFORMANCE CALIBRATION...' intact (an earlier version cut them at the colon and lost the real topic words).

**Lines 44-46**

```python
    cut = text.find(": ")
    if 0 < cut <= MAX_TITLE:
        return text[:cut].strip(), text[cut + 2:].strip()
```

Otherwise look for the first colon followed by a space. If it is within the first 190 characters, everything before it is the title and the rest is the body.

**Lines 47-48**

```python
    if len(text) <= MAX_TITLE + 40:
        return text.rstrip(": ").strip(), ""
```

If there is no suitable colon but the text is short, it is just a heading with no body.

**Line 49**

```python
    return text[:MAX_TITLE].rsplit(" ", 1)[0] + "...", text            # no clear heading: keep the whole paragraph as body
```

If nothing fits, the paragraph has no clear heading: the title is the first part of the text cut at a word boundary with '...', and the full paragraph stays as the body.

### Building the tree

**Lines 52-55**

```python
def build_tree(blocks: list[Block]) -> list[Section]:
    sections: list[Section] = [Section("0", "Front matter")]
    current_sub = Subsection("0.0", "Front matter")
    sections[0].subsections.append(current_sub)
```

`build_tree` starts with a 'section 0: Front matter' and an open subsection '0.0' to catch everything that appears before the first real section.

**Lines 57-58**

```python
    for block in blocks:
        if block.kind == "para":
```

Walks through every Block in order. Only paragraphs can be headings.

**Lines 59-60**

```python
            m = SECTION_RE.match(block.text)
            if m and len(block.text) <= MAX_TITLE + 60:
```

Checks whether the paragraph looks like a section heading (and is not absurdly long).

**Lines 61-65**

```python
                section = Section(m.group(1), m.group(2).strip().rstrip(":"))
                current_sub = Subsection(f"{section.number}.0", "Introduction")
                section.subsections.append(current_sub)
                sections.append(section)
                continue
```

If so: create a new Section (number and cleaned title), open an implicit 'n.0 Introduction' subsection for any text before the first numbered subsection, and move to the next Block.

**Lines 66-67**

```python
            m = SUB_RE.match(block.text)
            if m and m.group(1) == sections[-1].number:                        # n.m only counts inside section n
```

Otherwise checks for a subsection heading like `3.2 ...`. It only counts if its first number matches the current section (so a stray '7.3' inside section 1 is not a heading).

**Lines 68-70**

```python
                title, body = _split_title(f"{m.group(3)}")
                current_sub = Subsection(f"{m.group(1)}.{m.group(2)}", title)
                sections[-1].subsections.append(current_sub)
```

Splits the heading from its body, opens a new Subsection numbered '3.2', and attaches it to the current section.

**Lines 71-73**

```python
                if body:
                    current_sub.blocks.append(Block("para", text=body, page=block.page, meta=block.meta))
                continue
```

If the heading paragraph also carried body text, that body is saved as its own paragraph Block (keeping the page number), and the loop moves on.

**Line 74**

```python
        current_sub.blocks.append(block)
```

Any Block that is not a heading is added to the subsection that is currently open.

**Lines 76-77**

```python
    for section in sections:                                                    # drop implicit introductions with no content
        section.subsections = [s for s in section.subsections if s.blocks or not s.number.endswith(".0")]
```

After the loop: drop the implicit 'n.0 Introduction' subsections that stayed empty.

**Line 78**

```python
    return [s for s in sections if s.subsections]
```

Returns the list of sections that have at least one subsection.


---

## `agentic_rag/chunker.py`

Cuts each subsection into chunks. Different content gets different treatment: paragraphs are packed up to a size limit; the five repeated 'golden rule' paragraphs are always their own chunk; table rows are packed with the table title and column names repeated on every chunk (so a row never loses its meaning); figures become flagged placeholders.

### File description

**Lines 1-7**

```python
"""Chunking (plan section 5.3), done per subsection so a chunk never crosses a heading.

    paragraphs   packed up to ~doc_chunk_words; an over-long paragraph is cut on sentence boundaries
    rules        a "MANDATORY CROSS-DOMAIN RULE [R-02 ...]" paragraph is always its own chunk, tagged with its rule id
    tables       rows rendered as "Column: value | Column: value" and packed up to ~table_chunk_words; the table title and
                 column names are repeated on every chunk so a row is never separated from its meaning
    figures      kept as flagged placeholders (needs_vision) because their content is not in the text"""
```

Describes the four kinds of chunk and why. The key ideas: chunking is done per subsection so a chunk never crosses a heading, and tables repeat their title and column names so 'a row is never separated from its meaning'.

### Imports and a pattern

**Lines 9-11**

```python
import hashlib
import re
from dataclasses import dataclass, field
```

`hashlib` (makes fingerprints of text), `re` (patterns), and the record tools.

**Lines 13-16**

```python
from .blocks import Block
from .config import Config
from .structure import Subsection
from .text import sentences
```

Imports the pieces this file works with: `Block`, the `Config` settings, the `Subsection` record and the `sentences` helper.

**Line 18**

```python
RULE_ID = re.compile(r"\bR-0?(\d)\b")
```

`RULE_ID` finds rule codes like `R-02` or `R-2`: the letter R, a hyphen, an optional zero, and a digit, as a whole word (`\b` marks word edges).

### The DocChunk record

**Lines 21-22**

```python
@dataclass
class DocChunk:
```

A `DocChunk` is the output of this file: one piece of text ready to be stored.

**Lines 23-24**

```python
    kind: str                       # paragraph | rule | table | figure
    text: str
```

`kind` says which type it is (paragraph, rule, table or figure), and `text` is its content.

**Lines 25-26**

```python
    page_start: int | None = None
    page_end: int | None = None
```

`page_start` and `page_end`: the PDF pages it spans (nothing for Word and text files).

**Line 27**

```python
    needs_vision: bool = False
```

`needs_vision`: true for chunks whose real content is in an image that text extraction cannot read.

**Line 28**

```python
    meta: dict = field(default_factory=dict)
```

`meta`: a notebook for extras such as the rule id or the table's column names.

**Lines 30-33**

```python
    @property
    def content_hash(self) -> str:
        norm = re.sub(r"[^a-z0-9]+", " ", self.text.lower()).strip()
        return hashlib.sha1(norm.encode()).hexdigest()[:16]
```

`content_hash` is a computed property: a short fingerprint of the chunk's text. Letters are lower-cased and every non-alphanumeric run becomes a single space first, so the same text from the PDF, Word and text versions gets the **same** fingerprint, which lets duplicates across formats be found.

### Packing helpers

**Lines 36-37**

```python
def _pack(pieces: list[tuple[str, int | None]], limit: int) -> list[tuple[str, int | None, int | None]]:
    out, buf, words, p0, p1 = [], [], 0, None, None
```

`_pack` joins consecutive pieces of text into chunks that do not exceed a word limit. The state: finished chunks `out`, the chunk being built `buf`, its word count, and its first and last page.

**Lines 38-39**

```python
    for text, page in pieces:
        n = len(text.split())
```

For each piece: count its words.

**Lines 40-42**

```python
        if buf and words + n > limit:
            out.append((" ".join(buf), p0, p1))
            buf, words, p0, p1 = [], 0, None, None
```

If adding this piece would exceed the limit (and something is already in the buffer), the buffer is closed as a finished chunk and a new empty one starts.

**Lines 43-46**

```python
        buf.append(text)
        words += n
        p0 = page if p0 is None else p0
        p1 = page if page is not None else p1
```

Adds the piece to the buffer and updates the word count and the page range.

**Lines 47-49**

```python
    if buf:
        out.append((" ".join(buf), p0, p1))
    return out
```

After the loop, the last unfinished buffer is also stored. Returns all the chunks.

**Lines 52-65**

```python
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
```

`_pack_rows` does the same for table rows, but it returns **groups** of rows (lists), so the header can be repeated for each group later.

**Line 68**

```python
FIGURE_CAPTION = re.compile(r"^(\[CHART/FIGURE SUMMARY\]\s*)?Figure\s+\d", re.IGNORECASE)
```

`FIGURE_CAPTION` recognises a caption line such as 'Figure 1.1: ...' (optionally prefixed by `[CHART/FIGURE SUMMARY]`).

**Lines 71-73**

```python
def _split_long(text: str, limit: int, overlap: int = 0) -> list[str]:
    if len(text.split()) <= limit:
        return [text]
```

`_split_long` cuts one over-long paragraph. If it fits the limit it is returned as it is.

**Lines 74-76**

```python
    pieces = [t for t, _, _ in _pack([(s, None) for s in sentences(text)], limit)]
    if overlap <= 0:
        return pieces
```

Otherwise it is cut on **sentence boundaries** using `_pack` (never in the middle of a sentence). With no overlap requested, the pieces are returned.

**Lines 77-81**

```python
    out = [pieces[0]]
    for prev, piece in zip(pieces, pieces[1:]):
        tail = " ".join(sentences(prev)[-overlap:])
        out.append(f"{tail} {piece}")
    return out
```

With overlap, the last N sentences of each piece are repeated at the start of the next piece, so an idea that straddles a cut appears in both.

**Lines 84-85**

```python
def table_row_text(header: list[str], row: list[str]) -> str:
    return " | ".join(f"{h}: {c}" if h else c for h, c in zip(header, row) if c)
```

`table_row_text` writes one table row as text: 'Column: value | Column: value'. Empty cells are skipped and columns with no name show just the value.

### chunk_subsection(): the main function

**Lines 88-90**

```python
def chunk_subsection(sub: Subsection, config: Config) -> list[DocChunk]:
    chunks: list[DocChunk] = []
    pending: list[tuple[str, int | None]] = []
```

Takes a Subsection and the settings, and returns its chunks. `pending` collects paragraph pieces that are waiting to be packed together.

**Lines 92-95**

```python
    def flush() -> None:
        for text, p0, p1 in _pack(pending, config.doc_chunk_words):
            chunks.append(DocChunk("paragraph", text, p0, p1))
        pending.clear()
```

Inner helper `flush`: packs everything waiting in `pending` into paragraph chunks and empties the waiting list. It is called whenever something other than a normal paragraph is reached, so paragraphs on opposite sides of a table never share a chunk.

**Lines 97-98**

```python
    for i, block in enumerate(sub.blocks):
        if block.kind == "para":
```

Walks through the subsection's Blocks. (`enumerate` also gives the position `i`, used later to find neighbours.) Case 1: a paragraph.

**Lines 99-102**

```python
            if "MANDATORY CROSS-DOMAIN RULE" in block.text:
                flush()
                ids = sorted({f"R-0{d}" for d in RULE_ID.findall(block.text)})
                chunks.append(DocChunk("rule", block.text, block.page, block.page, meta={"rule_ids": ids}))
```

If the paragraph is one of the five 'golden rule' paragraphs: first `flush`, then store it as its own chunk of kind `rule`, tagged with the rule ids found in it (such as `R-02`).

**Lines 103-105**

```python
            else:
                for piece in _split_long(block.text, config.doc_chunk_words, config.chunk_overlap_sentences):
                    pending.append((piece, block.page))
```

Any other paragraph: cut it with `_split_long` if it is too long, and add the pieces to the waiting list.

**Lines 106-107**

```python
        elif block.kind == "table":
            flush()
```

Case 2: a table. First `flush` the waiting paragraphs.

**Lines 108-110**

```python
            rows = [table_row_text(block.header, r) for r in block.rows]
            last = block.meta.get("last_page", block.page)
            head = f"Table in {sub.number} {sub.title}. Columns: " + " | ".join(h for h in block.header if h) + "."
```

Renders each row as text, remembers the table's last page, and builds a heading line: 'Table in 4.3 Title. Columns: ...'.

**Lines 111-114**

```python
            groups = [[r] for r in rows] if config.table_row_chunks else _pack_rows(rows, config.table_chunk_words)
            for group in groups:
                chunks.append(DocChunk("table", head + " Rows:\n" + "\n".join(f"- {r}" for r in group), block.page, last,
                                       meta={"columns": block.header, "table_rows": len(block.rows)}))
```

Makes groups of rows: one row per group if row-level chunking is on, otherwise rows packed up to the size limit. Each group becomes a chunk: the heading line, then 'Rows:' and the rows as bullet lines. So the table title and columns are repeated in every chunk.

**Lines 115-116**

```python
        elif block.kind == "figure":
            flush()
```

Case 3: a figure. Flush first.

**Lines 117-119**

```python
            neighbours = [sub.blocks[j] for j in (i - 1, i + 1) if 0 <= j < len(sub.blocks)]
            caption = next((b.text for b in neighbours if b.kind == "para" and FIGURE_CAPTION.match(b.text)), "")
            text = (caption + " " if caption else "") + (block.text or "[Figure: an image with no extractable text]")
```

Looks at the Blocks directly before and after the figure. If one is a caption ('Figure 1.1: ...'), the caption is put in front of the figure's note.

**Lines 120-121**

```python
            chunks.append(DocChunk("figure", text, block.page, block.page, needs_vision=True,
                                   meta={**block.meta, "caption": caption}))
```

Stores a `figure` chunk marked `needs_vision`, with the caption in its notes.

**Lines 122-123**

```python
    flush()
    return chunks
```

After the last Block, flush any paragraphs still waiting, then return the chunks.


---
