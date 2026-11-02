"""A local, dependency-free embedder (hashed TF-IDF over stems, word pairs and character trigrams).

It is lexical, not neural: it handles typos and word forms but not deep paraphrase. That gap is exactly why branches carry
aliases (plan section 5.4). The Embedder interface is tiny, so a neural model can be swapped in later without touching
the rest of the pipeline."""

import math
import zlib
from collections import Counter

from .text import STOP, stem, tokenize

Vec = dict  # sparse vector: bucket -> weight


class HashEmbedder:
    kind = "local-hash-tfidf"

    def __init__(self, dim: int = 2048, idf: dict | None = None):
        self.dim = dim
        self.idf: dict[int, float] = {int(k): v for k, v in (idf or {}).items()}
        self.default_idf = max(self.idf.values()) if self.idf else 1.0

    def _features(self, text: str) -> dict:
        toks = [t for t in tokenize(text) if t not in STOP]
        stems = [stem(t) for t in toks]
        feats: dict[str, float] = {}

        def add(f: str, w: float) -> None:
            feats[f] = feats.get(f, 0.0) + w

        for s in stems:
            add("u:" + s, 1.0)
        for a, b in zip(stems, stems[1:]):
            add(f"b:{a}_{b}", 0.6)
        for t in toks:
            if len(t) >= 4 and t[0].isalpha():
                padded = f"#{t}#"
                for i in range(len(padded) - 2):
                    add("c:" + padded[i:i + 3], 0.15)
        return feats

    def _bucket(self, feature: str) -> int:
        return zlib.crc32(feature.encode()) % self.dim

    def fit(self, texts: list[str]) -> None:
        df: Counter = Counter()
        for t in texts:
            df.update({self._bucket(f) for f in self._features(t)})
        n = len(texts)
        self.idf = {b: math.log((n + 1) / (c + 1)) + 1.0 for b, c in df.items()}
        self.default_idf = math.log(n + 1) + 1.0

    def embed(self, text: str) -> Vec:
        vec: dict[int, float] = {}
        for f, w in self._features(text).items():
            b = self._bucket(f)
            vec[b] = vec.get(b, 0.0) + w * self.idf.get(b, self.default_idf)
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {b: v / norm for b, v in vec.items()}

    def to_json(self) -> dict:
        return {"kind": self.kind, "dim": self.dim, "idf": {str(k): v for k, v in self.idf.items()}}

    @classmethod
    def from_json(cls, data: dict) -> "HashEmbedder":
        return cls(dim=data["dim"], idf=data["idf"])


def cosine(a: Vec, b: Vec) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())
