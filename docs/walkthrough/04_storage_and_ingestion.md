# Part 4: Storing the chunks (SQLite, the in-memory store, ingestion)

Once documents are chunked, the chunks must be **stored** so they can be searched later without redoing the work. There are two storage layers. **SQLite** (`sqlite_store.py`) is the permanent copy: a database file holding documents, the section tree, chunks, vectors and a keyword index. The **in-memory `Store`** (`store.py`) is the working copy the question-answering pipeline uses: it is loaded from SQLite once and then filters what each user is allowed to see.

`ingest_docs.py` is the driver that runs the whole ingestion for real documents (load, structure, chunk, store, embed). `ingest.py` is an older, simpler ingestion for the small Markdown sample knowledge base that the first tests used.

## `agentic_rag/sqlite_store.py`

The permanent database. One file on disk, no server needed. It also contains a keyword search index (SQLite's built-in `FTS5`) which is the 'word-matching' half of hybrid search.

### File description

**Lines 1-10**

```python
"""SQLite storage for ingested documents (plan section 5.6).

Tables
    documents  one row per ingested file (format, hash, pages, counts)
    sections   the tree: level 1 = section (topic), level 2 = numbered subsection (type)
    chunks     the searchable text, with kind, page range, metadata (JSON), content hash and an embedding (sparse, JSON)
    chunks_fts SQLite FTS5 keyword index over chunk text (the "keyword half" of hybrid search)
    meta       embedder settings (IDF weights) so queries can be embedded the same way as the chunks

Re-ingesting a file with the same SHA-256 is skipped; a changed file replaces its old rows."""
```

The docstring lists the five tables and their purpose, and one rule: re-ingesting a file with the same SHA-256 fingerprint is skipped, while a changed file replaces its old rows.

### Imports

**Lines 12-16**

```python
import hashlib
import json
import sqlite3
import time
from pathlib import Path
```

`hashlib` (file fingerprints), `json` (store lists and dictionaries as text), `sqlite3` (the database), `time` (timestamps) and `Path`.

**Lines 18-23**

```python
from .chunker import DocChunk
from .config import Config
from .embeddings import HashEmbedder, embedder_from_json, make_embedder
from .models import Branch, Chunk
from .store import Store
from .structure import Section, Subsection
```

Imports the things this file stores or builds: chunk records, settings, the embedders, the data models, the in-memory `Store`, and the `Section`/`Subsection` records.

### The database schema

**Lines 25-28**

```python
SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, name TEXT, path TEXT, format TEXT, sha256 TEXT, bytes INTEGER, pages INTEGER,
    ingested_at TEXT, n_blocks INTEGER, n_sections INTEGER, n_chunks INTEGER);
```

`SCHEMA` is one long SQL text that creates the tables if they do not exist. The first, `documents`, has one row per ingested file: its id, name, path, format, SHA-256 fingerprint, size, page count, time of ingestion, and counts of blocks, sections and chunks.

**Lines 29-31**

```python
CREATE TABLE IF NOT EXISTS sections (
    id TEXT PRIMARY KEY, document_id TEXT REFERENCES documents(id), number TEXT, title TEXT, level INTEGER,
    parent_id TEXT, path TEXT, ordinal INTEGER, page_start INTEGER, page_end INTEGER, embedding TEXT, summary TEXT);
```

`sections` is the tree: one row per section (level 1) and per subsection (level 2), with its number, title, parent, full breadcrumb path, order, page range, its vector (`embedding`) and a short `summary` of its opening text. `REFERENCES documents(id)` ties each row to its document.

**Lines 32-35**

```python
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY, document_id TEXT REFERENCES documents(id), section_id TEXT REFERENCES sections(id),
    ordinal INTEGER, kind TEXT, text TEXT, context_prefix TEXT, page_start INTEGER, page_end INTEGER, words INTEGER,
    content_hash TEXT, needs_vision INTEGER DEFAULT 0, metadata TEXT, embedding TEXT);
```

`chunks` holds the searchable text: which document and section it belongs to, its order, its `kind` (paragraph / table / rule / figure), the text, the context prefix, pages, word count, the content fingerprint (`content_hash`), a flag for chunks needing OCR, the extra metadata (as JSON text), and its vector.

**Lines 36-38**

```python
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
```

Three indexes: lookup tables that make the database fast when searching chunks by document, by section, or by content fingerprint.

**Line 39**

```python
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, text, context_prefix, tokenize='porter unicode61');
```

`chunks_fts` is a **virtual table** using SQLite's FTS5 full-text search. It indexes the chunk text and prefix for word search with the 'porter' stemmer (so 'refunded' finds 'refund'). `chunk_id UNINDEXED` stores the id without indexing it.

**Lines 40-41**

```python
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""
```

`meta` is a small key/value table, used to remember which embedder made the vectors. The three quotes end the SQL text.

### Two small helpers

**Lines 44-45**

```python
def summarize_subsection(sub: Subsection, words: int = 45) -> str:
    """The first words of what a subsection says: its opening paragraph, or the column names of its first table."""
```

`summarize_subsection` makes the short summary stored with each section.

**Lines 46-48**

```python
    for block in sub.blocks:
        if block.kind == "para":
            return " ".join(block.text.split()[:words])
```

Walks through the subsection's Blocks. The first paragraph found is returned, cut to its first 45 words.

**Lines 49-50**

```python
        if block.kind == "table":
            return "Table of " + ", ".join(h for h in block.header if h)
```

If a table comes first, the summary is 'Table of' plus its column names.

**Line 51**

```python
    return ""
```

If there is neither, the summary is empty.

**Lines 54-55**

```python
def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
```

`sha256_of` reads a whole file and returns its SHA-256 fingerprint (a 64-character code that changes if even one byte of the file changes).

### Opening the database

**Lines 58-60**

```python
class SqliteStore:
    def __init__(self, path: str | Path = "data/documents.db"):
        self.path = Path(path)
```

The `SqliteStore` class wraps one database. The constructor takes a file path (default `data/documents.db`) and stores it as a `Path`.

**Lines 61-62**

```python
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
```

Creates the folder if it does not exist, then opens (or creates) the database file.

**Lines 63-64**

```python
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
```

`row_factory = Row` lets code read columns by name (`row["text"]`). `PRAGMA foreign_keys = ON` makes SQLite enforce the `REFERENCES` links.

**Line 65**

```python
        self.db.executescript(SCHEMA)
```

Runs the schema (creating any missing tables).

**Lines 66-67**

```python
        if "summary" not in {r["name"] for r in self.db.execute("PRAGMA table_info(sections)")}:
            self.db.execute("ALTER TABLE sections ADD COLUMN summary TEXT")      # databases created before summaries existed
```

A small migration: if the `sections` table was created by an older version without the `summary` column, the column is added.

**Lines 69-70**

```python
    def close(self) -> None:
        self.db.close()
```

`close` closes the database connection.

**Lines 72-75**

```python
    def reset(self) -> None:
        for table in ("chunks_fts", "chunks", "sections", "documents", "meta"):
            self.db.execute(f"DELETE FROM {table}")
        self.db.commit()
```

`reset` empties every table (the search index first), then saves with `commit`.

### Writing documents

**Lines 77-80**

```python
    # ---- writing ---------------------------------------------------------------------------------------------------
    def existing_hash(self, doc_id: str) -> str | None:
        row = self.db.execute("SELECT sha256 FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return row["sha256"] if row else None
```

`existing_hash` looks up a document id and returns its stored fingerprint, or `None` if not stored. It is how re-ingestion knows whether a file changed.

**Lines 82-86**

```python
    def delete_document(self, doc_id: str) -> None:
        self.db.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)", (doc_id,))
        self.db.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        self.db.execute("DELETE FROM sections WHERE document_id = ?", (doc_id,))
        self.db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
```

`delete_document` removes a document completely: its keyword-index rows, chunks, sections and the document row, in that order (children before parents, because of the `REFERENCES` links).

**Lines 88-91**

```python
    def add_document(self, doc_id: str, path: Path, fmt: str, pages: int | None, n_blocks: int,
                     tree: list[Section], chunked: dict[str, list[DocChunk]]) -> int:
        """Write one document in a single transaction. `chunked` maps a subsection number to its chunks."""
        n_chunks = sum(len(v) for v in chunked.values())
```

`add_document` writes one whole document. The docstring says `chunked` maps a subsection number to its chunks. The total number of chunks is counted first.

**Lines 92-93**

```python
        with self.db:
            self.delete_document(doc_id)
```

`with self.db:` starts a **transaction**: either everything below is saved, or (if anything fails) nothing is. It begins by deleting any older copy.

**Lines 94-97**

```python
            self.db.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (doc_id, path.name, str(path), fmt, sha256_of(path), path.stat().st_size, pages,
                 time.strftime("%Y-%m-%d %H:%M:%S"), n_blocks, len(tree), n_chunks))
```

Inserts the `documents` row: id, file name, path, format, fingerprint, size in bytes, page count, timestamp, and the counts.

**Lines 98-101**

```python
            order = 0
            for section in tree:
                sid = f"{doc_id}:{section.number}"
                s_path = f"Section {section.number}: {section.title}" if section.number != "0" else section.title
```

`order` numbers the sections in sequence. For each section: its id is `document:number`, and its path is 'Section 3: Title' (section 0 just uses its title).

**Lines 102-104**

```python
                self.db.execute("INSERT INTO sections (id, document_id, number, title, level, parent_id, path, ordinal) VALUES (?,?,?,?,?,?,?,?)",
                                (sid, doc_id, section.number, section.title, 1, None, s_path, order))
                order += 1
```

Inserts the section row (level 1, no parent) and increments the order counter.

**Lines 105-107**

```python
                for sub in section.subsections:
                    subid = f"{doc_id}:{sub.number}"
                    p0, p1 = sub.page_range()
```

For each subsection: its id, and its first and last page.

**Lines 108-112**

```python
                    self.db.execute("INSERT INTO sections (id, document_id, number, title, level, parent_id, path, ordinal, page_start, page_end, summary) "
                                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                    (subid, doc_id, sub.number, sub.title, 2, sid, f"{s_path} > {sub.number} {sub.title}",
                                     order, p0, p1, summarize_subsection(sub)))
                    order += 1
```

Inserts the subsection row (level 2) with its parent id, breadcrumb path (section path plus its own number and title), page range and summary.

**Lines 113-115**

```python
                    for i, ch in enumerate(chunked.get(sub.number, []), start=1):
                        cid = f"{subid}:{i:03d}"
                        prefix = f"{sub.number} {sub.title}:"
```

For each chunk of the subsection (numbered from 1): builds its id and the context prefix, such as '4.3 Records Retention:'.

**Lines 116-119**

```python
                        self.db.execute(
                            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (cid, doc_id, subid, i, ch.kind, ch.text, prefix, ch.page_start, ch.page_end,
                             len(ch.text.split()), ch.content_hash, int(ch.needs_vision), json.dumps(ch.meta), None))
```

Inserts the chunk row: the kind, text, prefix, pages, word count, fingerprint, vision flag and metadata (as JSON). The embedding is `None` for now; it is filled in later.

**Lines 120-121**

```python
                        self.db.execute("INSERT INTO chunks_fts VALUES (?,?,?)", (cid, ch.text, prefix))
        return n_chunks
```

Also puts the chunk into the keyword-search index, and returns the chunk count.

### Computing embeddings

**Lines 123-125**

```python
    def rebuild_embeddings(self, config: Config | None = None, kind: str = "hash") -> int:
        """(Re)compute every embedding with the chosen embedder ("neural" = MiniLM, "hash" = local TF-IDF). Run after any
        ingestion: the hash embedder's IDF weights depend on the whole corpus, and all vectors must come from one model."""
```

`rebuild_embeddings` turns every chunk and section into a vector. The docstring explains it must be re-run after any ingestion: all vectors have to come from one model, and the old hash model's weights depend on the whole collection.

**Lines 126-128**

```python
        config = config or Config()
        chunks = self.db.execute("SELECT id, text, context_prefix FROM chunks").fetchall()
        sections = self.db.execute("SELECT id, path, summary FROM sections").fetchall()
```

Uses default settings if none are given, and reads every chunk and every section from the database.

**Line 129**

```python
        embedder = make_embedder(kind, config.embedding_dim)
```

Creates the chosen embedder ('neural' or 'hash').

**Lines 130-131**

```python
        c_texts = [f"{c['context_prefix']} {c['text']}" for c in chunks]
        s_texts = [f"{s['path']} {s['summary'] or ''}" for s in sections]
```

Builds the text to embed: for a chunk, prefix plus text; for a section, its path plus its summary.

**Line 132**

```python
        embedder.fit(c_texts + s_texts)
```

Lets the embedder learn from all the texts (`fit`). The neural model does nothing here; the hash model learns word weights.

**Lines 133-134**

```python
        many = embedder.embed_many if hasattr(embedder, "embed_many") else (lambda ts: [embedder.embed(t) for t in ts])
        enc = lambda vec: json.dumps({str(k): round(v, 5) for k, v in vec.items()})
```

`many` is the embedder's batch method if it has one (the neural model is faster in batches), otherwise a simple loop. `enc` converts a vector to a JSON string with numbers rounded to 5 decimals.

**Lines 135-137**

```python
        with self.db:
            for c, vec in zip(chunks, many(c_texts)):
                self.db.execute("UPDATE chunks SET embedding = ? WHERE id = ?", (enc(vec), c["id"]))
```

In one transaction: for each chunk, saves its vector.

**Lines 138-139**

```python
            for s, vec in zip(sections, many(s_texts)):
                self.db.execute("UPDATE sections SET embedding = ? WHERE id = ?", (enc(vec), s["id"]))
```

The same for each section.

**Lines 140-141**

```python
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('embedder', ?)", (json.dumps(embedder.to_json()),))
        return len(chunks)
```

Remembers which embedder was used in the `meta` table (so queries later use the same one) and returns the number of chunks.

### Reading from the database

**Lines 143-145**

```python
    # ---- reading -----------------------------------------------------------------------------------------------------
    def keyword_search(self, query: str, limit: int = 5) -> list[sqlite3.Row]:
        """FTS5 keyword search (BM25 ranking). Each word is quoted so punctuation cannot break the query."""
```

`keyword_search` runs a word search using the FTS5 index (ranked by BM25, a standard word-relevance formula).

**Lines 146-148**

```python
        terms = " OR ".join(f'"{t}"' for t in "".join(c if c.isalnum() else " " for c in query).split())
        if not terms:
            return []
```

Turns the question into a safe search expression: every non-letter becomes a space, then each word is put in quotes and joined with OR. The quoting stops punctuation in the question from breaking the query. An empty question returns nothing.

**Lines 149-152**

```python
        return self.db.execute(
            "SELECT c.id, c.document_id, c.kind, c.page_start, substr(c.text, 1, 160) AS preview, bm25(chunks_fts) AS score "
            "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.chunk_id WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
            (terms, limit)).fetchall()
```

The SQL: join the index with the chunks table to get the chunk details and a 160-character preview, keep only matches, order by relevance score, and return up to `limit` rows.

**Lines 154-156**

```python
    def load_store(self) -> Store:
        """Build the in-memory Store the Pipeline works on, from what is in SQLite. Every document is kept as its own
        branch tree; branch ids carry the document id."""
```

`load_store` builds the in-memory `Store` the pipeline works on. Each document keeps its own tree, and branch ids contain the document id.

**Lines 157-159**

```python
        meta = self.db.execute("SELECT value FROM meta WHERE key = 'embedder'").fetchone()
        if meta is None:
            raise RuntimeError("no embeddings stored yet: run rebuild_embeddings() first")
```

Reads which embedder was used. If none is stored, embeddings were never built, so it stops with a clear error.

**Lines 160-162**

```python
        dec = lambda text: {int(k): v for k, v in json.loads(text).items()} if text else {}
        docs = {r["id"]: r["name"] for r in self.db.execute("SELECT id, name FROM documents")}
        branches, chunks = [], []
```

`dec` converts a stored vector text back into a dictionary. Then it reads the document names and prepares two lists.

**Lines 163-165**

```python
        for r in self.db.execute("SELECT * FROM sections ORDER BY document_id, ordinal"):
            branches.append(Branch(id=r["id"], name=r["title"], level=r["level"], parent_id=r["parent_id"], path=r["path"],
                                   description=r["summary"] or "", aliases=[], embedding=dec(r["embedding"]), access=["public"]))
```

Turns every section row into a `Branch`, using the summary as its description. All branches are `public` here.

**Lines 166-168**

```python
        for r in self.db.execute("SELECT * FROM chunks ORDER BY document_id, section_id, ordinal"):
            chunks.append(Chunk(id=r["id"], branch_id=r["section_id"], text=r["text"], context_prefix=r["context_prefix"],
                                source=docs[r["document_id"]], embedding=dec(r["embedding"])))
```

Turns every chunk row into a `Chunk`, with the document name as its source.

**Line 169**

```python
        return Store(embedder_from_json(json.loads(meta["value"])), branches, chunks)
```

Returns the `Store`, with the same embedder that made the vectors.

### Statistics

**Lines 171-173**

```python
    def stats(self) -> dict:
        q = lambda sql, *a: self.db.execute(sql, a).fetchall()
        docs = []
```

`stats` produces the numbers shown by `db-info`. `q` is a tiny helper that runs a query and returns all rows.

**Lines 174-176**

```python
        for d in q("SELECT * FROM documents ORDER BY name, format"):
            kinds = {r["kind"]: r["n"] for r in q("SELECT kind, COUNT(*) n FROM chunks WHERE document_id = ? GROUP BY kind", d["id"])}
            w = q("SELECT MIN(words) lo, CAST(AVG(words) AS INT) av, MAX(words) hi, SUM(words) total FROM chunks WHERE document_id = ?", d["id"])[0]
```

For each document: counts its chunks by kind, and gets the minimum, average, maximum and total chunk size in words.

**Lines 177-182**

```python
            docs.append({"id": d["id"], "name": d["name"], "format": d["format"], "pages": d["pages"], "blocks": d["n_blocks"],
                         "sections": q("SELECT COUNT(*) n FROM sections WHERE document_id = ? AND level = 1", d["id"])[0]["n"],
                         "subsections": q("SELECT COUNT(*) n FROM sections WHERE document_id = ? AND level = 2", d["id"])[0]["n"],
                         "chunks": d["n_chunks"], "kinds": kinds, "words": dict(w),
                         "needs_vision": q("SELECT COUNT(*) n FROM chunks WHERE document_id = ? AND needs_vision = 1", d["id"])[0]["n"],
                         "embedded": q("SELECT COUNT(*) n FROM chunks WHERE document_id = ? AND embedding IS NOT NULL", d["id"])[0]["n"]})
```

Builds one summary row per document: format, pages, blocks, number of sections and subsections, chunk counts, chunk sizes, how many chunks need OCR, and how many have a vector.

**Line 183**

```python
        hashes = {d["id"]: {r["content_hash"] for r in q("SELECT content_hash FROM chunks WHERE document_id = ?", d["id"])} for d in docs}
```

Collects, for each document, the set of its chunk fingerprints.

**Lines 184-188**

```python
        agreement = {}
        for a in docs:
            for b in docs:
                if a["id"] != b["id"] and hashes[a["id"]]:
                    agreement[(a["format"], b["format"])] = len(hashes[a["id"]] & hashes[b["id"]]) / len(hashes[a["id"]])
```

For every pair of different documents: the share of chunks in the first whose fingerprint also appears in the second. This is the 'chunk agreement across formats' figure.

**Lines 189-190**

```python
        return {"documents": docs, "agreement": agreement,
                "db_bytes": self.path.stat().st_size, "fts_rows": q("SELECT COUNT(*) n FROM chunks_fts")[0]["n"]}
```

Returns the documents, the agreement figures, the database size, and the number of rows in the keyword index.


---

## `agentic_rag/ingest_docs.py`

The ingestion driver for PDF, Word and text files: it runs load, structure, chunk, store and embed for each file, skipping files that have not changed.

### File description and imports

**Line 1**

```python
"""Orchestrates document ingestion: load -> structure -> chunk -> SQLite -> embeddings (plan section 5)."""
```

Names the steps the file orchestrates: load, structure, chunk, SQLite, embeddings.

**Lines 3-4**

```python
import time
from pathlib import Path
```

`time` (for timing) and `Path`.

**Lines 6-11**

```python
from .chunker import chunk_subsection
from .config import Config
from .loaders import load_document
from .sqlite_store import SqliteStore, sha256_of
from .structure import build_tree
from .text import slug
```

Imports one function from each previous stage, so this file simply calls them in order.

### A stable id for each file

**Lines 14-15**

```python
def doc_id_for(path: Path) -> str:
    return f"{slug(path.stem)}-{path.suffix.lower().lstrip('.')}"
```

`doc_id_for` builds a document id from the file name and extension, for example 'Rag testing doc.pdf' becomes `rag-testing-doc-pdf`. The same file therefore always gets the same id, which is what lets re-ingestion detect changes.

### ingest_files(): the main loop

**Lines 18-19**

```python
def ingest_files(paths: list[str | Path], db_path: str | Path = "data/documents.db", config: Config | None = None,
                 reset: bool = False, log=print, embedder: str = "hash") -> SqliteStore:
```

Takes a list of files, the database path, optional settings, a `reset` flag, a logging function (default `print`), and the embedder name.

**Lines 20-24**

```python
    config = config or Config()
    store = SqliteStore(db_path)
    if reset:
        store.reset()
    changed = False
```

Uses default settings if none are given, opens the database, empties it if `reset` was requested, and starts with `changed = False`.

**Lines 25-27**

```python
    for raw in paths:
        path = Path(raw)
        doc_id = doc_id_for(path)
```

For each file: build its id.

**Lines 28-30**

```python
        if store.existing_hash(doc_id) == sha256_of(path):
            log(f"  skip   {path.name}: unchanged since the last ingestion")
            continue
```

If a document with this id is already stored **and** its fingerprint is the same, nothing changed: print 'skip' and go to the next file.

**Line 31**

```python
        t0 = time.time()
```

Starts a timer.

**Lines 32-34**

```python
        blocks = load_document(path)
        tree = build_tree(blocks)
        chunked = {sub.number: chunk_subsection(sub, config) for sec in tree for sub in sec.subsections}
```

The core of ingestion in three lines: load the file into Blocks; build the tree of sections; chunk every subsection of every section.

**Line 35**

```python
        pages = max((b.meta.get("last_page", b.page) or 0 for b in blocks), default=0) or None
```

Works out the page count: the highest page number among the Blocks (or `None` for files without pages).

**Line 36**

```python
        n = store.add_document(doc_id, path, path.suffix.lower().lstrip("."), pages, len(blocks), tree, chunked)
```

Stores the document, with its tree and chunks, in SQLite.

**Lines 37-38**

```python
        log(f"  loaded {path.name}: {len(blocks)} blocks -> {len(tree)} sections -> {n} chunks ({time.time() - t0:.1f}s)")
        changed = True
```

Prints a summary line (blocks, sections, chunks, seconds) and notes that something changed.

**Lines 39-41**

```python
    if changed:
        n = store.rebuild_embeddings(config, embedder)
        log(f"  embedded {n} chunks with the {embedder} embedder")
```

After the loop: if any file was new or changed, recompute **all** embeddings (they must all come from one model and one learning pass) and report how many chunks were embedded.

**Line 42**

```python
    return store
```

Returns the open store.

### Printing the statistics

**Lines 45-48**

```python
def format_stats(stats: dict) -> str:
    lines = ["", f"SQLite database: {stats['db_bytes'] / 1024:.0f} KB, {stats['fts_rows']} rows in the keyword index", ""]
    head = f"{'document':<40}{'fmt':<6}{'pages':>6}{'sect':>6}{'subs':>6}{'chunks':>8}   chunk kinds"
    lines += [head, "-" * len(head)]
```

`format_stats` turns the statistics into readable text. It starts with the database size and a table header (document, format, pages, sections, subsections, chunks, chunk kinds).

**Lines 49-51**

```python
    for d in stats["documents"]:
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(d["kinds"].items()))
        lines.append(f"{d['name'][:39]:<40}{d['format']:<6}{d['pages'] or '-':>6}{d['sections']:>6}{d['subsections']:>6}{d['chunks']:>8}   {kinds}")
```

One table row per document, with the chunk kinds listed alphabetically.

**Lines 52-56**

```python
    lines += ["", "Chunk size in words (min / avg / max) and embedding coverage:"]
    for d in stats["documents"]:
        w = d["words"]
        lines.append(f"  {d['format']:<5} {w['lo']} / {w['av']} / {w['hi']}   total {w['total']} words   embedded {d['embedded']}/{d['chunks']}"
                     + (f"   {d['needs_vision']} figure chunk(s) need OCR/vision" if d["needs_vision"] else ""))
```

Then the chunk size in words (min / avg / max), the total words and how many chunks are embedded, with a warning if any chunks need OCR.

**Lines 57-60**

```python
    if stats["agreement"]:
        lines += ["", "Chunk agreement across formats (share of A's chunks whose text is identical to a chunk in B):"]
        for (a, b), v in sorted(stats["agreement"].items()):
            lines.append(f"  {a:>5} -> {b:<5} {v * 100:5.1f}%")
```

If there is more than one document, it prints the chunk agreement between every pair of formats as a percentage.

**Line 61**

```python
    return "\n".join(lines)
```

Joins all the lines into one block of text.


---

## `agentic_rag/store.py`

The in-memory working copy. It holds every branch and chunk, and answers 'what is this user allowed to see?' and 'what words exist in what they can see?'. Permission filtering lives **here, in code**, and is never left to a prompt that a model could be tricked into ignoring.

### File description and imports

**Line 1**

```python
"""In-memory vector store with permission and status filtering. Persisted as one JSON file."""
```

Describes the file: an in-memory store with permission and status filtering, savable to one JSON file.

**Lines 3-5**

```python
import json
from collections import Counter
from pathlib import Path
```

`json`, a `Counter` (counts things) and `Path`.

**Lines 7-9**

```python
from .embeddings import HashEmbedder, cosine, embedder_from_json
from .models import Branch, Chunk
from .text import content_stems, content_tokens
```

Imports the embedder type, the cosine-similarity function and the embedder factory; the Branch and Chunk records; and the text helpers.

### The Store: holding the data

**Lines 12-14**

```python
class Store:
    def __init__(self, embedder: HashEmbedder, branches: list[Branch], chunks: list[Chunk]):
        self.embedder = embedder
```

A `Store` is created from an embedder, a list of branches and a list of chunks.

**Lines 15-17**

```python
        self.branches = {b.id: b for b in branches}
        self.chunks = chunks
        self._vocab_cache: dict = {}
```

Branches are kept in a dictionary keyed by id for fast lookup; chunks stay a list; `_vocab_cache` will remember vocabulary calculations so they are done only once per user group.

### What the user is allowed to see

**Lines 19-22**

```python
    # ---- visibility (plan 6.2): enforced in code, never left to a prompt --------------------------------------
    @staticmethod
    def _allowed(access: list[str], groups) -> bool:
        return bool(set(access) & set(groups))
```

`_allowed` is the permission test: true if the item's access list and the user's groups share at least one group (`&` is set intersection).

**Lines 24-25**

```python
    def visible_branches(self, groups) -> list[Branch]:
        return [b for b in self.branches.values() if b.status == "active" and self._allowed(b.access, groups)]
```

`visible_branches`: every branch that is active (not retired) and permitted.

**Lines 27-28**

```python
    def topics(self, groups) -> list[Branch]:
        return [b for b in self.visible_branches(groups) if b.level == 1]
```

`topics`: the visible level-1 branches (sections).

**Lines 30-31**

```python
    def leaves(self, groups) -> list[Branch]:
        return [b for b in self.visible_branches(groups) if b.level == 2]
```

`leaves`: the visible level-2 branches (subsections), which are the ones that hold text.

**Lines 33-35**

```python
    def chunks_for(self, branch_id: str, groups) -> list[Chunk]:
        return [c for c in self.chunks
                if c.branch_id == branch_id and c.status == "active" and self._allowed(c.access, groups)]
```

`chunks_for`: the chunks of one branch that are active and permitted.

**Lines 37-40**

```python
    def visible_chunks(self, groups) -> list[Chunk]:
        return [c for c in self.chunks
                if c.status == "active" and self._allowed(c.access, groups)
                and self.branches[c.branch_id].status == "active"]
```

`visible_chunks`: every chunk the user may see, which is also active and belongs to an active branch. Retired policies can never be returned this way.

### Vocabulary of what the user can see

**Lines 42-44**

```python
    # ---- vocabulary of what this user is allowed to see (used for typo repair and "unknown term" detection) ----
    def vocab(self, groups) -> tuple[set, Counter, int]:
        key = tuple(sorted(groups))
```

`vocab` returns the words present in the visible content. It is used to repair typos and to detect question words the knowledge base has never seen (like 'car' in a policy manual). The cache key is the user's groups.

**Lines 45-48**

```python
        if key not in self._vocab_cache:
            raw: set[str] = set()
            df: Counter = Counter()
            docs = 0
```

If not computed yet: prepares a set of raw words, a counter of how many chunks contain each word stem (`df`, 'document frequency'), and a document count.

**Lines 49-53**

```python
            for c in self.visible_chunks(groups):
                text = c.embed_text()
                raw.update(content_tokens(text))
                df.update(set(content_stems(text)))
                docs += 1
```

For every visible chunk: add its words to the raw set, add each distinct word stem to the counter, and count the chunk.

**Lines 54-57**

```python
            for b in self.visible_branches(groups):
                text = b.search_text()
                raw.update(content_tokens(text))
                df.update(set(content_stems(text)))
```

The same for every visible branch's search text (but branches do not add to the chunk count).

**Lines 58-59**

```python
            self._vocab_cache[key] = (raw, df, max(docs, 1))
        return self._vocab_cache[key]
```

Saves and returns (raw words, stem counts, document count; at least 1). Because it only looks at **visible** content, a staff-only word never leaks to a public user.

### Simple search, saving and loading

**Lines 61-65**

```python
    # ---- search ------------------------------------------------------------------------------------------------
    def search_chunks(self, qvec: dict, groups, k: int) -> list[tuple[Chunk, float]]:
        scored = [(c, cosine(qvec, c.embedding)) for c in self.visible_chunks(groups)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
```

`search_chunks`: scores every visible chunk by cosine similarity to the question vector, sorts best first, and returns the top `k`. (This is the basic search used by the older ranking and the older fallback.)

**Lines 67-71**

```python
    # ---- persistence -------------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        def enc(vec: dict) -> dict:
            return {str(k): round(v, 6) for k, v in vec.items()}
```

`save` writes the store to a JSON file. `enc` rounds vector numbers to 6 decimals to keep the file small.

**Lines 72-78**

```python
        data = {
            "embedder": self.embedder.to_json(),
            "branches": [{**b.__dict__, "embedding": enc(b.embedding)} for b in self.branches.values()],
            "chunks": [{**c.__dict__, "embedding": enc(c.embedding)} for c in self.chunks],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data))
```

Builds a dictionary of the embedder's settings, every branch and every chunk (their fields via `__dict__`, with rounded vectors), creates the folder if needed, and writes the JSON.

**Lines 80-85**

```python
    @classmethod
    def load(cls, path: str | Path) -> "Store":
        data = json.loads(Path(path).read_text())

        def dec(vec: dict) -> dict:
            return {int(k): v for k, v in vec.items()}
```

`load` is a class method that reads the JSON file back. `dec` turns text keys back into numbers (JSON keys are always text).

**Lines 87-89**

```python
        branches = [Branch(**{**b, "embedding": dec(b["embedding"])}) for b in data["branches"]]
        chunks = [Chunk(**{**c, "embedding": dec(c["embedding"])}) for c in data["chunks"]]
        return cls(embedder_from_json(data["embedder"]), branches, chunks)
```

Rebuilds the Branch and Chunk records, and the embedder that made them (neural or hash), and returns a new `Store`.


---

## `agentic_rag/ingest.py`

The older, simpler ingestion for the small sample knowledge base: Markdown files with a header block, one file per topic. It is kept because the first tests and the regression set use it. It builds the same Branch and Chunk records as the document ingestion, but straight from Markdown instead of from PDF or Word.

### File description and imports

**Lines 1-18**

```python
"""Offline half of the pipeline (plan section 5): load -> structure -> chunk -> enrich -> embed -> store.

Knowledge-base format (one Markdown file per topic):

    ---
    topic: Refund
    access: public            # comma separated groups
    status: active            # active | retired
    version: 3                # only needed to keep a retired copy next to the active one
    aliases: money back; reimbursement
    description: Rules for getting money back for an order.
    source: Returns_Policy
    ---
    ## Damage refund
    aliases: broken item; arrived smashed
    description: Refunds for items that arrive broken.
    The text of the section follows as ordinary paragraphs.
"""
```

The docstring explains the Markdown format: a `---` header with topic, access, status, version, aliases, description and source; then `## Type name` sections, each with optional `aliases:` and `description:` lines followed by ordinary paragraphs.

**Lines 20-26**

```python
from pathlib import Path

from .config import Config
from .embeddings import HashEmbedder
from .models import Branch, Chunk
from .store import Store
from .text import sentences, slug
```

Imports the path tool, the settings, the hash embedder, the records, the store, and the text helpers.

**Line 28**

```python
META_KEYS = ("aliases", "description", "access", "status")
```

`META_KEYS`: the lines inside a section that are settings, not text.

### Small helpers

**Lines 31-32**

```python
def _split_list(value: str, sep: str = ";") -> list[str]:
    return [v.strip() for v in value.split(sep) if v.strip()]
```

`_split_list` splits a text like 'a; b; c' into a clean list ['a', 'b', 'c'], dropping empty pieces. The separator defaults to `;`.

### Parsing one Markdown file

**Lines 35-37**

```python
def parse_document(path: Path) -> tuple[dict, list[dict]]:
    """Return (front_matter, sections). Each section: {name, meta, text}."""
    raw = path.read_text().splitlines()
```

`parse_document` returns the header values and a list of sections. It reads the file as lines.

**Lines 38-39**

```python
    if not raw or raw[0].strip() != "---":
        raise ValueError(f"{path}: missing front matter")
```

A file that does not start with `---` is rejected with a clear error.

**Lines 40-45**

```python
    end = raw[1:].index("---") + 1
    front = {}
    for line in raw[1:end]:
        if ":" in line:
            k, v = line.split(":", 1)
            front[k.strip().lower()] = v.split("  #")[0].strip()
```

Finds the closing `---` and reads each `key: value` line of the header into a dictionary (keys lower-cased; anything after a `  #` comment marker is dropped).

**Lines 46-47**

```python
    sections: list[dict] = []
    current = None
```

Prepares the section list and a pointer to the section being read.

**Lines 48-51**

```python
    for line in raw[end + 1:]:
        if line.startswith("## "):
            current = {"name": line[3:].strip(), "meta": {}, "lines": []}
            sections.append(current)
```

For each line after the header: a line starting `## ` begins a new section named by the rest of the line.

**Lines 52-55**

```python
        elif current is not None:
            key = line.split(":", 1)[0].strip().lower()
            if ":" in line and key in META_KEYS:
                current["meta"][key] = line.split(":", 1)[1].strip()
```

Other lines inside a section: if the line is a `key: value` whose key is one of the setting names, it is stored as that section's setting.

**Lines 56-57**

```python
            else:
                current["lines"].append(line)
```

Otherwise it is ordinary text and is kept in the section's list of lines.

**Lines 58-59**

```python
    for s in sections:
        s["text"] = "\n".join(s.pop("lines")).strip()
```

Afterwards, each section's lines are joined into one text and trimmed.

**Line 60**

```python
    return front, sections
```

Returns the header and the sections.

### Chunking the sample text

**Lines 63-65**

```python
def chunk_text(text: str, max_words: int) -> list[str]:
    """Pack paragraphs, then sentences, into chunks of roughly max_words, without cutting a sentence in half."""
    pieces: list[str] = []
```

`chunk_text` is the simple chunker for the sample knowledge base: it packs paragraphs up to about `max_words` without cutting a sentence in half.

**Lines 66-67**

```python
    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        pieces.extend(sentences(para) if len(para.split()) > max_words else [para])
```

Splits the text into paragraphs. A paragraph longer than the limit is replaced by its sentences; a shorter paragraph stays whole.

**Lines 68-75**

```python
    chunks, buf, count = [], [], 0
    for piece in pieces:
        n = len(piece.split())
        if buf and count + n > max_words:
            chunks.append(" ".join(buf))
            buf, count = [], 0
        buf.append(piece)
        count += n
```

Packs the pieces into chunks: a new chunk starts whenever the next piece would push the word count over the limit.

**Lines 76-78**

```python
    if buf:
        chunks.append(" ".join(buf))
    return chunks
```

Stores the last chunk and returns all of them.

### build_store(): files to an in-memory store

**Lines 81-84**

```python
def build_store(kb_dir: str | Path, config: Config | None = None) -> Store:
    config = config or Config()
    branches: dict[str, Branch] = {}
    chunks: list[Chunk] = []
```

Takes a folder and builds a `Store`. `branches` is a dictionary keyed by id; `chunks` is a list.

**Lines 86-88**

```python
    for path in sorted(Path(kb_dir).glob("*.md")):
        front, sections = parse_document(path)
        topic_name = front["topic"]
```

For each `.md` file in the folder (sorted so the order is stable): parse it and read the topic name.

**Lines 89-93**

```python
        access = _split_list(front.get("access", "public"), ",")
        status = front.get("status", "active")
        version = front.get("version", "1")
        source = front.get("source", path.stem)
        topic_id = slug(topic_name)
```

Reads the shared settings: access groups (comma-separated), status, version and source name; the topic id is a slug of its name.

**Lines 95-99**

```python
        if topic_id not in branches:
            branches[topic_id] = Branch(
                id=topic_id, name=topic_name, level=1, parent_id=None, path=topic_name,
                description=front.get("description", ""), aliases=_split_list(front.get("aliases", "")),
                access=access, status="active")
```

If this topic is new, creates a topic Branch (level 1) with its description, aliases and access.

**Lines 100-103**

```python
        else:  # a second file for the same topic only adds to its aliases and widens access
            existing = branches[topic_id]
            existing.aliases += [a for a in _split_list(front.get("aliases", "")) if a not in existing.aliases]
            existing.access = sorted(set(existing.access) | set(access))
```

If another file already created this topic, the new file only adds its aliases and widens the access list. This is how a retired old version of a topic file sits next to the active one.

**Lines 105-108**

```python
        for sec in sections:
            suffix = f"@{version}" if status != "active" else ""
            bid = f"{topic_id}.{slug(sec['name'])}{suffix}"
            meta = sec["meta"]
```

For each section of the file: its id is topic + slug of its name (with `@version` added for non-active files, so a retired copy gets its own id).

**Lines 109-114**

```python
            branch = Branch(
                id=bid, name=sec["name"], level=2, parent_id=topic_id, path=f"{topic_name} > {sec['name']}",
                description=meta.get("description", ""), aliases=_split_list(meta.get("aliases", "")),
                access=_split_list(meta["access"], ",") if "access" in meta else access,
                status=meta.get("status", status), version=version)
            branches[bid] = branch
```

Creates the type Branch (level 2): its path, description, aliases, access (from the section's own line if given, else the file's), status and version. It is added to the dictionary.

**Lines 115-118**

```python
            for i, piece in enumerate(chunk_text(sec["text"], config.chunk_size_words), 1):
                chunks.append(Chunk(
                    id=f"{bid}.{i:03d}", branch_id=bid, text=piece, context_prefix=f"{branch.path}:",
                    source=source, access=branch.access, status=branch.status))
```

Chunks the section text and creates a Chunk for each piece, with the branch path as its prefix and the branch's access and status.

**Lines 120-121**

```python
    embedder = HashEmbedder(dim=config.embedding_dim)
    embedder.fit([b.search_text() for b in branches.values()] + [c.embed_text() for c in chunks])
```

Creates the embedder and lets it learn from all branch texts and chunk texts.

**Lines 122-125**

```python
    for b in branches.values():
        b.embedding = embedder.embed(b.search_text())
    for c in chunks:
        c.embedding = embedder.embed(c.embed_text())
```

Gives every branch and every chunk its vector.

**Line 126**

```python
    return Store(embedder, list(branches.values()), chunks)
```

Returns the finished in-memory `Store`.


---
