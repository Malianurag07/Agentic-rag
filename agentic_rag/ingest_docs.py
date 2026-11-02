"""Orchestrates document ingestion: load -> structure -> chunk -> SQLite -> embeddings (plan section 5)."""

import time
from pathlib import Path

from .chunker import chunk_subsection
from .config import Config
from .loaders import load_document
from .sqlite_store import SqliteStore, sha256_of
from .structure import build_tree
from .text import slug


def doc_id_for(path: Path) -> str:
    return f"{slug(path.stem)}-{path.suffix.lower().lstrip('.')}"


def ingest_files(paths: list[str | Path], db_path: str | Path = "data/documents.db", config: Config | None = None,
                 reset: bool = False, log=print, embedder: str = "hash") -> SqliteStore:
    config = config or Config()
    store = SqliteStore(db_path)
    if reset:
        store.reset()
    changed = False
    for raw in paths:
        path = Path(raw)
        doc_id = doc_id_for(path)
        if store.existing_hash(doc_id) == sha256_of(path):
            log(f"  skip   {path.name}: unchanged since the last ingestion")
            continue
        t0 = time.time()
        blocks = load_document(path)
        tree = build_tree(blocks)
        chunked = {sub.number: chunk_subsection(sub, config) for sec in tree for sub in sec.subsections}
        pages = max((b.meta.get("last_page", b.page) or 0 for b in blocks), default=0) or None
        n = store.add_document(doc_id, path, path.suffix.lower().lstrip("."), pages, len(blocks), tree, chunked)
        log(f"  loaded {path.name}: {len(blocks)} blocks -> {len(tree)} sections -> {n} chunks ({time.time() - t0:.1f}s)")
        changed = True
    if changed:
        n = store.rebuild_embeddings(config, embedder)
        log(f"  embedded {n} chunks with the {embedder} embedder")
    return store


def format_stats(stats: dict) -> str:
    lines = ["", f"SQLite database: {stats['db_bytes'] / 1024:.0f} KB, {stats['fts_rows']} rows in the keyword index", ""]
    head = f"{'document':<40}{'fmt':<6}{'pages':>6}{'sect':>6}{'subs':>6}{'chunks':>8}   chunk kinds"
    lines += [head, "-" * len(head)]
    for d in stats["documents"]:
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(d["kinds"].items()))
        lines.append(f"{d['name'][:39]:<40}{d['format']:<6}{d['pages'] or '-':>6}{d['sections']:>6}{d['subsections']:>6}{d['chunks']:>8}   {kinds}")
    lines += ["", "Chunk size in words (min / avg / max) and embedding coverage:"]
    for d in stats["documents"]:
        w = d["words"]
        lines.append(f"  {d['format']:<5} {w['lo']} / {w['av']} / {w['hi']}   total {w['total']} words   embedded {d['embedded']}/{d['chunks']}"
                     + (f"   {d['needs_vision']} figure chunk(s) need OCR/vision" if d["needs_vision"] else ""))
    if stats["agreement"]:
        lines += ["", "Chunk agreement across formats (share of A's chunks whose text is identical to a chunk in B):"]
        for (a, b), v in sorted(stats["agreement"].items()):
            lines.append(f"  {a:>5} -> {b:<5} {v * 100:5.1f}%")
    return "\n".join(lines)
