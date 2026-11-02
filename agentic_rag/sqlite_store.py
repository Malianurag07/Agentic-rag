"""SQLite storage for ingested documents (plan section 5.6).

Tables
    documents  one row per ingested file (format, hash, pages, counts)
    sections   the tree: level 1 = section (topic), level 2 = numbered subsection (type)
    chunks     the searchable text, with kind, page range, metadata (JSON), content hash and an embedding (sparse, JSON)
    chunks_fts SQLite FTS5 keyword index over chunk text (the "keyword half" of hybrid search)
    meta       embedder settings (IDF weights) so queries can be embedded the same way as the chunks

Re-ingesting a file with the same SHA-256 is skipped; a changed file replaces its old rows."""

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from .chunker import DocChunk
from .config import Config
from .embeddings import HashEmbedder
from .models import Branch, Chunk
from .store import Store
from .structure import Section, Subsection

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, name TEXT, path TEXT, format TEXT, sha256 TEXT, bytes INTEGER, pages INTEGER,
    ingested_at TEXT, n_blocks INTEGER, n_sections INTEGER, n_chunks INTEGER);
CREATE TABLE IF NOT EXISTS sections (
    id TEXT PRIMARY KEY, document_id TEXT REFERENCES documents(id), number TEXT, title TEXT, level INTEGER,
    parent_id TEXT, path TEXT, ordinal INTEGER, page_start INTEGER, page_end INTEGER, embedding TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY, document_id TEXT REFERENCES documents(id), section_id TEXT REFERENCES sections(id),
    ordinal INTEGER, kind TEXT, text TEXT, context_prefix TEXT, page_start INTEGER, page_end INTEGER, words INTEGER,
    content_hash TEXT, needs_vision INTEGER DEFAULT 0, metadata TEXT, embedding TEXT);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, text, context_prefix, tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def summarize_subsection(sub: Subsection, words: int = 45) -> str:
    """The first words of what a subsection says: its opening paragraph, or the column names of its first table."""
    for block in sub.blocks:
        if block.kind == "para":
            return " ".join(block.text.split()[:words])
        if block.kind == "table":
            return "Table of " + ", ".join(h for h in block.header if h)
    return ""


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SqliteStore:
    def __init__(self, path: str | Path = "data/documents.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        if "summary" not in {r["name"] for r in self.db.execute("PRAGMA table_info(sections)")}:
            self.db.execute("ALTER TABLE sections ADD COLUMN summary TEXT")      # databases created before summaries existed

    def close(self) -> None:
        self.db.close()

    def reset(self) -> None:
        for table in ("chunks_fts", "chunks", "sections", "documents", "meta"):
            self.db.execute(f"DELETE FROM {table}")
        self.db.commit()

    # ---- writing ---------------------------------------------------------------------------------------------------
    def existing_hash(self, doc_id: str) -> str | None:
        row = self.db.execute("SELECT sha256 FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return row["sha256"] if row else None

    def delete_document(self, doc_id: str) -> None:
        self.db.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)", (doc_id,))
        self.db.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        self.db.execute("DELETE FROM sections WHERE document_id = ?", (doc_id,))
        self.db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def add_document(self, doc_id: str, path: Path, fmt: str, pages: int | None, n_blocks: int,
                     tree: list[Section], chunked: dict[str, list[DocChunk]]) -> int:
        """Write one document in a single transaction. `chunked` maps a subsection number to its chunks."""
        n_chunks = sum(len(v) for v in chunked.values())
        with self.db:
            self.delete_document(doc_id)
            self.db.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (doc_id, path.name, str(path), fmt, sha256_of(path), path.stat().st_size, pages,
                 time.strftime("%Y-%m-%d %H:%M:%S"), n_blocks, len(tree), n_chunks))
            order = 0
            for section in tree:
                sid = f"{doc_id}:{section.number}"
                s_path = f"Section {section.number}: {section.title}" if section.number != "0" else section.title
                self.db.execute("INSERT INTO sections (id, document_id, number, title, level, parent_id, path, ordinal) VALUES (?,?,?,?,?,?,?,?)",
                                (sid, doc_id, section.number, section.title, 1, None, s_path, order))
                order += 1
                for sub in section.subsections:
                    subid = f"{doc_id}:{sub.number}"
                    p0, p1 = sub.page_range()
                    self.db.execute("INSERT INTO sections (id, document_id, number, title, level, parent_id, path, ordinal, page_start, page_end, summary) "
                                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                    (subid, doc_id, sub.number, sub.title, 2, sid, f"{s_path} > {sub.number} {sub.title}",
                                     order, p0, p1, summarize_subsection(sub)))
                    order += 1
                    for i, ch in enumerate(chunked.get(sub.number, []), start=1):
                        cid = f"{subid}:{i:03d}"
                        prefix = f"{sub.number} {sub.title}:"
                        self.db.execute(
                            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (cid, doc_id, subid, i, ch.kind, ch.text, prefix, ch.page_start, ch.page_end,
                             len(ch.text.split()), ch.content_hash, int(ch.needs_vision), json.dumps(ch.meta), None))
                        self.db.execute("INSERT INTO chunks_fts VALUES (?,?,?)", (cid, ch.text, prefix))
        return n_chunks

    def rebuild_embeddings(self, config: Config | None = None) -> int:
        """(Re)compute every embedding. IDF weights depend on the whole corpus, so this runs after any ingestion."""
        config = config or Config()
        chunks = self.db.execute("SELECT id, text, context_prefix FROM chunks").fetchall()
        sections = self.db.execute("SELECT id, path, summary FROM sections").fetchall()
        embedder = HashEmbedder(dim=config.embedding_dim)
        embedder.fit([f"{c['context_prefix']} {c['text']}" for c in chunks] + [f"{s['path']} {s['summary'] or ''}" for s in sections])
        with self.db:
            for c in chunks:
                vec = embedder.embed(f"{c['context_prefix']} {c['text']}")
                self.db.execute("UPDATE chunks SET embedding = ? WHERE id = ?", (json.dumps({str(k): round(v, 5) for k, v in vec.items()}), c["id"]))
            for s in sections:
                vec = embedder.embed(f"{s['path']} {s['summary'] or ''}")
                self.db.execute("UPDATE sections SET embedding = ? WHERE id = ?", (json.dumps({str(k): round(v, 5) for k, v in vec.items()}), s["id"]))
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('embedder', ?)", (json.dumps(embedder.to_json()),))
        return len(chunks)

    # ---- reading -----------------------------------------------------------------------------------------------------
    def keyword_search(self, query: str, limit: int = 5) -> list[sqlite3.Row]:
        """FTS5 keyword search (BM25 ranking). Each word is quoted so punctuation cannot break the query."""
        terms = " OR ".join(f'"{t}"' for t in "".join(c if c.isalnum() else " " for c in query).split())
        if not terms:
            return []
        return self.db.execute(
            "SELECT c.id, c.document_id, c.kind, c.page_start, substr(c.text, 1, 160) AS preview, bm25(chunks_fts) AS score "
            "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.chunk_id WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
            (terms, limit)).fetchall()

    def load_store(self) -> Store:
        """Build the in-memory Store the Pipeline works on, from what is in SQLite. Every document is kept as its own
        branch tree; branch ids carry the document id."""
        meta = self.db.execute("SELECT value FROM meta WHERE key = 'embedder'").fetchone()
        if meta is None:
            raise RuntimeError("no embeddings stored yet: run rebuild_embeddings() first")
        dec = lambda text: {int(k): v for k, v in json.loads(text).items()} if text else {}
        docs = {r["id"]: r["name"] for r in self.db.execute("SELECT id, name FROM documents")}
        branches, chunks = [], []
        for r in self.db.execute("SELECT * FROM sections ORDER BY document_id, ordinal"):
            branches.append(Branch(id=r["id"], name=r["title"], level=r["level"], parent_id=r["parent_id"], path=r["path"],
                                   description=r["summary"] or "", aliases=[], embedding=dec(r["embedding"]), access=["public"]))
        for r in self.db.execute("SELECT * FROM chunks ORDER BY document_id, section_id, ordinal"):
            chunks.append(Chunk(id=r["id"], branch_id=r["section_id"], text=r["text"], context_prefix=r["context_prefix"],
                                source=docs[r["document_id"]], embedding=dec(r["embedding"])))
        return Store(HashEmbedder.from_json(json.loads(meta["value"])), branches, chunks)

    def stats(self) -> dict:
        q = lambda sql, *a: self.db.execute(sql, a).fetchall()
        docs = []
        for d in q("SELECT * FROM documents ORDER BY name, format"):
            kinds = {r["kind"]: r["n"] for r in q("SELECT kind, COUNT(*) n FROM chunks WHERE document_id = ? GROUP BY kind", d["id"])}
            w = q("SELECT MIN(words) lo, CAST(AVG(words) AS INT) av, MAX(words) hi, SUM(words) total FROM chunks WHERE document_id = ?", d["id"])[0]
            docs.append({"id": d["id"], "name": d["name"], "format": d["format"], "pages": d["pages"], "blocks": d["n_blocks"],
                         "sections": q("SELECT COUNT(*) n FROM sections WHERE document_id = ? AND level = 1", d["id"])[0]["n"],
                         "subsections": q("SELECT COUNT(*) n FROM sections WHERE document_id = ? AND level = 2", d["id"])[0]["n"],
                         "chunks": d["n_chunks"], "kinds": kinds, "words": dict(w),
                         "needs_vision": q("SELECT COUNT(*) n FROM chunks WHERE document_id = ? AND needs_vision = 1", d["id"])[0]["n"],
                         "embedded": q("SELECT COUNT(*) n FROM chunks WHERE document_id = ? AND embedding IS NOT NULL", d["id"])[0]["n"]})
        hashes = {d["id"]: {r["content_hash"] for r in q("SELECT content_hash FROM chunks WHERE document_id = ?", d["id"])} for d in docs}
        agreement = {}
        for a in docs:
            for b in docs:
                if a["id"] != b["id"] and hashes[a["id"]]:
                    agreement[(a["format"], b["format"])] = len(hashes[a["id"]] & hashes[b["id"]]) / len(hashes[a["id"]])
        return {"documents": docs, "agreement": agreement,
                "db_bytes": self.path.stat().st_size, "fts_rows": q("SELECT COUNT(*) n FROM chunks_fts")[0]["n"]}
