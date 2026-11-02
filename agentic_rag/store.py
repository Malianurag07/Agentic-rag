"""In-memory vector store with permission and status filtering. Persisted as one JSON file."""

import json
from collections import Counter
from pathlib import Path

from .embeddings import HashEmbedder, cosine
from .models import Branch, Chunk
from .text import content_stems, content_tokens


class Store:
    def __init__(self, embedder: HashEmbedder, branches: list[Branch], chunks: list[Chunk]):
        self.embedder = embedder
        self.branches = {b.id: b for b in branches}
        self.chunks = chunks
        self._vocab_cache: dict = {}

    # ---- visibility (plan 6.2): enforced in code, never left to a prompt --------------------------------------
    @staticmethod
    def _allowed(access: list[str], groups) -> bool:
        return bool(set(access) & set(groups))

    def visible_branches(self, groups) -> list[Branch]:
        return [b for b in self.branches.values() if b.status == "active" and self._allowed(b.access, groups)]

    def topics(self, groups) -> list[Branch]:
        return [b for b in self.visible_branches(groups) if b.level == 1]

    def leaves(self, groups) -> list[Branch]:
        return [b for b in self.visible_branches(groups) if b.level == 2]

    def chunks_for(self, branch_id: str, groups) -> list[Chunk]:
        return [c for c in self.chunks
                if c.branch_id == branch_id and c.status == "active" and self._allowed(c.access, groups)]

    def visible_chunks(self, groups) -> list[Chunk]:
        return [c for c in self.chunks
                if c.status == "active" and self._allowed(c.access, groups)
                and self.branches[c.branch_id].status == "active"]

    # ---- vocabulary of what this user is allowed to see (used for typo repair and "unknown term" detection) ----
    def vocab(self, groups) -> tuple[set, Counter, int]:
        key = tuple(sorted(groups))
        if key not in self._vocab_cache:
            raw: set[str] = set()
            df: Counter = Counter()
            docs = 0
            for c in self.visible_chunks(groups):
                text = c.embed_text()
                raw.update(content_tokens(text))
                df.update(set(content_stems(text)))
                docs += 1
            for b in self.visible_branches(groups):
                text = b.search_text()
                raw.update(content_tokens(text))
                df.update(set(content_stems(text)))
            self._vocab_cache[key] = (raw, df, max(docs, 1))
        return self._vocab_cache[key]

    # ---- search ------------------------------------------------------------------------------------------------
    def search_chunks(self, qvec: dict, groups, k: int) -> list[tuple[Chunk, float]]:
        scored = [(c, cosine(qvec, c.embedding)) for c in self.visible_chunks(groups)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ---- persistence -------------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        def enc(vec: dict) -> dict:
            return {str(k): round(v, 6) for k, v in vec.items()}

        data = {
            "embedder": self.embedder.to_json(),
            "branches": [{**b.__dict__, "embedding": enc(b.embedding)} for b in self.branches.values()],
            "chunks": [{**c.__dict__, "embedding": enc(c.embedding)} for c in self.chunks],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "Store":
        data = json.loads(Path(path).read_text())

        def dec(vec: dict) -> dict:
            return {int(k): v for k, v in vec.items()}

        branches = [Branch(**{**b, "embedding": dec(b["embedding"])}) for b in data["branches"]]
        chunks = [Chunk(**{**c, "embedding": dec(c["embedding"])}) for c in data["chunks"]]
        return cls(HashEmbedder.from_json(data["embedder"]), branches, chunks)
