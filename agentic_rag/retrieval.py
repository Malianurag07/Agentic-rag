"""Retrieval (plan section 6.3): find the chunks that hold the answer, then the sections they belong to.

    dense     meaning match: cosine similarity between the question and chunk embeddings
    bm25      word match: classic BM25 over the chunk text and its section title (exact terms, codes, numbers)
    hybrid    both, merged by reciprocal rank fusion (a chunk ranked well by either list rises)

Sections are ranked by their best chunk, so a fact buried deep in a section's body still lifts that section."""

import math
import re
from collections import Counter, defaultdict

from .embeddings import cosine
from .models import Chunk
from .text import content_stems


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(docs)) if docs else 1.0
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for t in set(query):
                f = tf.get(t, 0)
                if f:
                    s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder, mode: str = "hybrid", rrf_k: int = 60, dense_weight: float = 1.0):
        self.chunks, self.embedder, self.mode = chunks, embedder, mode
        self.rrf_k, self.dense_weight = rrf_k, dense_weight
        self.bm25 = BM25([content_stems(c.embed_text()) for c in chunks])

    def _dense(self, q: str) -> list[float]:
        qv = self.embedder.embed(q)
        return [cosine(qv, c.embedding) for c in self.chunks]

    @staticmethod
    def _ranks(scores: list[float]) -> dict[int, int]:
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return {i: r for r, i in enumerate(order, start=1)}

    def rank_chunks(self, q: str, k: int = 10) -> list[tuple[Chunk, float]]:
        n = len(self.chunks)
        dense = self._dense(q) if self.mode in ("dense", "hybrid") else [0.0] * n
        sparse = self.bm25.scores(content_stems(q)) if self.mode in ("bm25", "hybrid") else [0.0] * n
        if self.mode == "dense":
            fused = dense
        elif self.mode == "bm25":
            fused = sparse
        else:
            rd, rs = self._ranks(dense), self._ranks(sparse)
            fused = [self.dense_weight / (self.rrf_k + rd[i]) + (1.0 / (self.rrf_k + rs[i]) if sparse[i] > 0 else 0.0) for i in range(n)]
        order = sorted(range(n), key=lambda i: fused[i], reverse=True)[:k]
        return [(self.chunks[i], fused[i]) for i in order]

    def rank_chunks_detailed(self, q: str, k: int = 40) -> list[tuple[Chunk, float, float, int, int]]:
        """(chunk, fused score, dense cosine, dense rank, bm25 rank) best-first. The dense cosine is a meaningful absolute
        number with a neural embedder (relevant is high, unrelated is low); the two ranks show whether the rankers agree."""
        n = len(self.chunks)
        dense, sparse = self._dense(q), self.bm25.scores(content_stems(q))
        rd, rs = self._ranks(dense), self._ranks(sparse)
        fused = [self.dense_weight / (self.rrf_k + rd[i]) + (1.0 / (self.rrf_k + rs[i]) if sparse[i] > 0 else 0.0) for i in range(n)]
        order = sorted(range(n), key=lambda i: fused[i], reverse=True)[:k]
        return [(self.chunks[i], fused[i], dense[i], rd[i], rs[i] if sparse[i] > 0 else n + 1) for i in order]

    @staticmethod
    def rank_sections(ranked: list[tuple[Chunk, float]]) -> list[tuple[str, float]]:
        best: dict[str, float] = {}
        for chunk, score in ranked:
            best.setdefault(chunk.branch_id, score)             # chunks arrive best-first, so the first one is the section's best
        return list(best.items())


def make_units(chunk: Chunk, window: int = 2, include_whole: bool = True) -> list[str]:
    """Small units to MATCH on: each table row (with the table title and columns), or short sentence windows of a paragraph.
    A fact in one row of a long table, or in one sentence of a long paragraph, is no longer diluted by its neighbours."""
    from .text import sentences
    prefix = chunk.context_prefix
    if "Rows:\n" in chunk.text:
        head, _, body = chunk.text.partition("Rows:\n")
        units = [f"{prefix} {head.strip()} {r[2:] if r.startswith('- ') else r}" for r in body.split("\n") if r.strip()]
    else:
        sents = sentences(chunk.text)
        units = [f"{prefix} {' '.join(sents[i:i + window])}" for i in range(0, max(1, len(sents) - window + 1))] if len(sents) > window else []
    if include_whole or not units:
        units.append(chunk.embed_text())
    return units


class SmallToBigRetriever:
    """Match on small units, return their parent chunks (plan 5.3, parent-child chunking). A parent is ranked by its best unit."""

    def __init__(self, chunks: list[Chunk], embedder, mode: str = "hybrid", window: int = 2, include_whole: bool = True,
                 dense_weight: float = 1.0, rrf_k: int = 60):
        self.parents = chunks
        units, owner = [], []
        for i, c in enumerate(chunks):
            for text in make_units(c, window, include_whole):
                units.append(text)
                owner.append(i)
        vecs = embedder.embed_many(units) if hasattr(embedder, "embed_many") else [embedder.embed(t) for t in units]
        self.unit_chunks = [Chunk(id=f"u{j}", branch_id=chunks[owner[j]].branch_id, text=units[j], context_prefix="", embedding=vecs[j])
                            for j in range(len(units))]
        self.owner = owner
        self.inner = Retriever(self.unit_chunks, embedder, mode=mode, dense_weight=dense_weight, rrf_k=rrf_k)

    def _parents(self, ranked_units, k):
        out, seen = [], set()
        for unit, *rest in ranked_units:
            j = int(unit.id[1:])
            parent = self.parents[self.owner[j]]
            if parent.id in seen:
                continue
            seen.add(parent.id)
            out.append((parent, *rest))
            if len(out) >= k:
                break
        return out

    def rank_chunks(self, q: str, k: int = 10):
        return self._parents(self.inner.rank_chunks(q, k=k * 6), k)

    def rank_chunks_detailed(self, q: str, k: int = 40):
        return self._parents(self.inner.rank_chunks_detailed(q, k=k * 6), k)

    @staticmethod
    def rank_sections(ranked):
        return Retriever.rank_sections(ranked)
