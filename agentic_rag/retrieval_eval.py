"""Judge-free retrieval evaluation: does the retriever put the answer in front of the rest of the pipeline?

For each answerable question (one that has expected facts) and each document format it measures
    section@k   is a section that holds the answer among the top k sections?  (multi-part: every needed section)
    facts@k     do the top k chunks, together, contain every expected fact?
No evidence check, chooser or answer writer is involved, so this isolates chunking, embeddings and retrieval."""

import json
import re
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from .chunker import chunk_subsection
from .config import Config
from .embeddings import HashEmbedder
from .loaders import load_document
from .models import Chunk
from .pipeline import Pipeline
from .retrieval import Retriever, SmallToBigRetriever
from .structure import build_tree
from .text import slug

NORM = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
ROOT = Path(__file__).resolve().parent.parent
FILES = [ROOT / "Rag testing doc.pdf", ROOT / "enterprise_company_policy_manual.docx", ROOT / "enterprise_company_policy_manual.txt"]
_neural = None


@lru_cache(maxsize=None)
def _tree(path: str):
    return build_tree(load_document(path))


def get_embedder(kind: str, texts: list[str]):
    global _neural
    if kind == "neural":
        if _neural is None:
            from .neural import MiniLMEmbedder
            _neural = MiniLMEmbedder()
        return _neural
    emb = HashEmbedder(dim=2048)
    emb.fit(texts)
    return emb


@lru_cache(maxsize=None)
def corpus(path: str, config_key: tuple, embedder_kind: str):
    """Chunks (with embeddings) for one document under one chunking configuration."""
    options = dict(config_key)
    prefix_mode = options.pop("prefix", "sub")
    config = replace(Config(), **options)
    doc_id = f"{slug(Path(path).stem)}-{Path(path).suffix.lstrip('.')}"
    chunks: list[Chunk] = []
    for sec in _tree(path):
        for sub in sec.subsections:
            prefix = (f"{sub.number} {sub.title}:" if prefix_mode == "sub"
                      else f"Section {sec.number}: {sec.title} > {sub.number} {sub.title}:")
            for i, ch in enumerate(chunk_subsection(sub, config), start=1):
                chunks.append(Chunk(id=f"{doc_id}:{sub.number}:{i:03d}", branch_id=f"{doc_id}:{sub.number}", text=ch.text, context_prefix=prefix))
    texts = [c.embed_text() for c in chunks]
    emb = get_embedder(embedder_kind, texts)
    vecs = emb.embed_many(texts) if hasattr(emb, "embed_many") else [emb.embed(t) for t in texts]
    for c, v in zip(chunks, vecs):
        c.embedding = v
    return emb, chunks


def load_questions(*paths) -> list[dict]:
    return [json.loads(l) for p in paths for l in Path(p).read_text().splitlines() if l.strip()]


def _query(q: dict) -> str:
    return (q["history"][-1] + " " if q.get("history") else "") + q["question"]


def evaluate(questions: list[dict], chunk_overrides: dict, embedder_kind: str, mode: str, split: bool = False,
             dense_weight: float = 1.0, rrf_k: int = 60, small_to_big: dict | None = None) -> dict:
    key = tuple(sorted(chunk_overrides.items()))
    answerable = [q for q in questions if q.get("contains") and q["status"] != "refused"]
    hits = {f"section@{k}": 0 for k in (1, 3, 5)} | {f"facts@{k}": 0 for k in (3, 5, 10)}
    n = 0
    for path in FILES:
        emb, chunks = corpus(str(path), key, embedder_kind)
        retriever = (SmallToBigRetriever(chunks, emb, mode=mode, dense_weight=dense_weight, rrf_k=rrf_k, **small_to_big)
                     if small_to_big is not None else Retriever(chunks, emb, mode=mode, dense_weight=dense_weight, rrf_k=rrf_k))
        for q in answerable:
            query = _query(q)
            subs = Pipeline._plan(query) if split else [query]
            ranked_lists = [retriever.rank_chunks(s, k=20) for s in subs]
            if len(ranked_lists) == 1:
                ranked = ranked_lists[0]
            else:                                                             # interleave the sub-questions' results
                ranked, seen = [], set()
                for rank in range(20):
                    for lst in ranked_lists:
                        if rank < len(lst) and lst[rank][0].id not in seen:
                            seen.add(lst[rank][0].id)
                            ranked.append(lst[rank])
            sections = [s.split(":", 1)[1] for s, _ in retriever.rank_sections(ranked)]
            reqs = q.get("sections", [])
            sat = lambda pool: all(any(o in pool for o in (r if isinstance(r, list) else [r])) for r in reqs)
            for k in (1, 3, 5):
                hits[f"section@{k}"] += sat(set(sections[:k]))
            facts = [NORM(f) for f in q["contains"]]
            for k in (3, 5, 10):
                text = NORM(" ".join(c.text for c, _ in ranked[:k]))
                hits[f"facts@{k}"] += all(f in text for f in facts)
            n += 1
    return {k: v / n for k, v in hits.items()} | {"n": n}
