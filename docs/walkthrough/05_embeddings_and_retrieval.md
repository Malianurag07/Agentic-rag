# Part 5: Embeddings and retrieval (turning text into numbers, then finding the right chunks)

A computer cannot compare meanings directly, but it can compare numbers. An **embedding** is a list of numbers that represents a piece of text, built so that texts with similar meaning get similar numbers. Retrieval then becomes arithmetic: turn the question into numbers, and find the chunks whose numbers are closest.

There are two embedders. The **hash embedder** (`embeddings.py`) is a simple word-counting scheme that needs nothing installed. The **neural embedder** (`neural.py`) is a small trained model that understands meaning, so 'how long are chat messages kept?' lands near 'Ephemeral Messaging: 180 days'. `retrieval.py` then combines meaning-matching with word-matching (BM25), merges the two rankings, and optionally matches on small pieces while returning the bigger chunk they came from.

## `agentic_rag/embeddings.py`

The simple, dependency-free embedder, the similarity function used by everything, and two small factory functions that choose or rebuild an embedder.

### File description and imports

**Lines 1-5**

```python
"""A local, dependency-free embedder (hashed TF-IDF over stems, word pairs and character trigrams).

It is lexical, not neural: it handles typos and word forms but not deep paraphrase. That gap is exactly why branches carry
aliases (plan section 5.4). The Embedder interface is tiny, so a neural model can be swapped in later without touching
the rest of the pipeline."""
```

Explains that this embedder is *lexical* (word-based), not neural: it copes with typos and word forms but not with real paraphrase. That gap is why branches carry aliases. It also notes the interface is tiny, so a neural model can replace it without changing the rest of the pipeline (which is exactly what `neural.py` later does).

**Lines 7-9**

```python
import math
import zlib
from collections import Counter
```

`math` (logarithms and square roots), `zlib` (a fast checksum used as a hash), and `Counter`.

**Line 11**

```python
from .text import STOP, stem, tokenize
```

Imports the stop-word list, the stemmer and the tokenizer from `text.py`.

**Line 13**

```python
Vec = dict  # sparse vector: bucket -> weight
```

`Vec` is just a name for a dictionary used as a **sparse vector**: it maps a position number to a value, and only non-zero positions are stored.

### The HashEmbedder

**Lines 16-17**

```python
class HashEmbedder:
    kind = "local-hash-tfidf"
```

The class. `kind` is a label saved with the vectors so the right embedder can be rebuilt later.

**Lines 19-22**

```python
    def __init__(self, dim: int = 2048, idf: dict | None = None):
        self.dim = dim
        self.idf: dict[int, float] = {int(k): v for k, v in (idf or {}).items()}
        self.default_idf = max(self.idf.values()) if self.idf else 1.0
```

The constructor takes the vector size (`dim`, 2048 slots) and optionally previously learned word weights (`idf`). It converts the stored weights' keys back to numbers, and sets a default weight for words never seen before (the largest weight, or 1.0 if nothing has been learned yet).

**Lines 24-27**

```python
    def _features(self, text: str) -> dict:
        toks = [t for t in tokenize(text) if t not in STOP]
        stems = [stem(t) for t in toks]
        feats: dict[str, float] = {}
```

`_features` turns text into a bag of features with weights. It tokenizes, drops stop words, and stems what is left. `feats` collects the weights.

**Lines 29-30**

```python
        def add(f: str, w: float) -> None:
            feats[f] = feats.get(f, 0.0) + w
```

Inner helper `add`: add a weight to a feature, starting from 0 if it is new.

**Lines 32-33**

```python
        for s in stems:
            add("u:" + s, 1.0)
```

Feature type 1: each **stemmed word**, weight 1.0 (prefixed `u:` for 'unigram').

**Lines 34-35**

```python
        for a, b in zip(stems, stems[1:]):
            add(f"b:{a}_{b}", 0.6)
```

Feature type 2: each **pair of neighbouring words**, weight 0.6 (prefixed `b:`). This lets 'credit card' count as more than 'credit' plus 'card'.

**Lines 36-40**

```python
        for t in toks:
            if len(t) >= 4 and t[0].isalpha():
                padded = f"#{t}#"
                for i in range(len(padded) - 2):
                    add("c:" + padded[i:i + 3], 0.15)
```

Feature type 3: **three-letter slices of each word**, weight 0.15 (prefixed `c:`). The word is padded with `#` at both ends so the first and last slices are marked, for example '#re', 'ref', 'efu'... This lets a misspelling such as 'refnd' still overlap with 'refund'. Only words of four or more letters that start with a letter are sliced.

**Line 41**

```python
        return feats
```

Returns the features.

**Lines 43-44**

```python
    def _bucket(self, feature: str) -> int:
        return zlib.crc32(feature.encode()) % self.dim
```

`_bucket` maps a feature name to a position between 0 and `dim`-1 using a checksum (`crc32`) and the remainder. This is called **hashing**: no dictionary of all words has to be kept.

**Lines 46-49**

```python
    def fit(self, texts: list[str]) -> None:
        df: Counter = Counter()
        for t in texts:
            df.update({self._bucket(f) for f in self._features(t)})
```

`fit` learns how common each position is across all the texts: for every text, the set of positions it uses is counted (`df`, document frequency).

**Lines 50-52**

```python
        n = len(texts)
        self.idf = {b: math.log((n + 1) / (c + 1)) + 1.0 for b, c in df.items()}
        self.default_idf = math.log(n + 1) + 1.0
```

Then each position gets an IDF weight, `log((n+1)/(count+1)) + 1`: rare positions weigh more, common ones less. A default weight is set for unseen positions.

**Lines 54-58**

```python
    def embed(self, text: str) -> Vec:
        vec: dict[int, float] = {}
        for f, w in self._features(text).items():
            b = self._bucket(f)
            vec[b] = vec.get(b, 0.0) + w * self.idf.get(b, self.default_idf)
```

`embed` turns one text into a vector: for each feature, add its weight times its position's IDF weight into the vector.

**Lines 59-60**

```python
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {b: v / norm for b, v in vec.items()}
```

Divides every value by the vector's length (**normalisation**), so every vector has length 1. Then the similarity of two texts is just a sum of products.

**Lines 62-63**

```python
    def to_json(self) -> dict:
        return {"kind": self.kind, "dim": self.dim, "idf": {str(k): v for k, v in self.idf.items()}}
```

`to_json` produces the settings to save in the database: the label, size and learned weights.

**Lines 65-67**

```python
    @classmethod
    def from_json(cls, data: dict) -> "HashEmbedder":
        return cls(dim=data["dim"], idf=data["idf"])
```

`from_json` rebuilds an embedder from those saved settings.

### Similarity and the factories

**Lines 70-73**

```python
def cosine(a: Vec, b: Vec) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())
```

`cosine` measures how alike two vectors are. Because both are normalised, it is simply the sum of products over positions they share. For speed it loops over the smaller vector and looks values up in the bigger one. 1.0 means identical, 0 means nothing in common.

**Lines 76-81**

```python
def embedder_from_json(data: dict):
    """Rebuild the embedder a database was created with (the stored vectors only make sense with the same model)."""
    if data.get("kind") == "onnx-minilm-l6-v2":
        from .neural import MiniLMEmbedder
        return MiniLMEmbedder.from_json(data)
    return HashEmbedder.from_json(data)
```

`embedder_from_json` rebuilds whichever embedder created a stored database. A database made with the neural model must be queried with the neural model, or the numbers would mean nothing. The import is inside the `if` so the neural libraries load only when needed.

**Lines 84-88**

```python
def make_embedder(kind: str, dim: int = 2048):
    if kind == "neural":
        from .neural import MiniLMEmbedder
        return MiniLMEmbedder()
    return HashEmbedder(dim=dim)
```

`make_embedder` creates a new embedder by name: 'neural' gives the MiniLM model; anything else gives the hash embedder.


---

## `agentic_rag/neural.py`

The neural embedder: a small trained model (**all-MiniLM-L6-v2**, in the ONNX format) that runs on an ordinary CPU. It turns any text of up to about 200 words into 384 numbers whose closeness reflects meaning. It also contains the safe download code for the model file.

### File description and imports

**Lines 1-7**

```python
"""A neural sentence-embedding model (all-MiniLM-L6-v2, ONNX) that understands meaning, not just shared words.

"How long are chat messages kept?" and "Ephemeral Messaging: 180 Calendar Days" share almost no words, but their embeddings are close.
Runs on CPU with onnxruntime. The model file (about 83 MB) is downloaded once by `python -m agentic_rag download-model`; its
SHA-256 is checked against the value published in ChromaDB's source, which distributes the same file.

Inputs longer than MAX_TOKENS word-pieces are cut off, so chunks should stay below roughly 150 words."""
```

Describes what it does with an example (a question and a table row that share almost no words but have close embeddings). The model file is about 83 MB, is downloaded once, and its SHA-256 fingerprint is checked against the value published in ChromaDB's source. Inputs over 256 word-pieces are cut off, which is why chunks should stay shorter than roughly 150 words.

**Lines 9-13**

```python
import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path
```

`hashlib` (fingerprint), `os` (environment variables), `tarfile` (unpack the download), `urllib.request` (download) and `Path`.

**Line 15**

```python
import numpy as np
```

`numpy` is the library for fast arithmetic on arrays of numbers. It does the maths of combining the model's outputs.

### Constants

**Line 17**

```python
MODEL_URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
```

`MODEL_URL`: where the model is downloaded from.

**Line 18**

```python
MODEL_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
```

`MODEL_SHA256`: the exact fingerprint the downloaded file must have. If even one byte differs, the file is rejected. This protects against a corrupted or tampered download.

**Line 19**

```python
DEFAULT_DIR = Path(os.environ.get("AGENTIC_RAG_MODEL_DIR", Path(__file__).resolve().parent.parent / "models" / "all-MiniLM-L6-v2"))
```

`DEFAULT_DIR`: where the model lives on disk: the folder named by the `AGENTIC_RAG_MODEL_DIR` environment variable if set, otherwise `models/all-MiniLM-L6-v2` in the project.

**Line 20**

```python
MAX_TOKENS = 256
```

`MAX_TOKENS`: the model reads at most 256 word-pieces per text.

### Downloading the model safely

**Lines 23-24**

```python
def download_model(target: Path = DEFAULT_DIR, log=print) -> Path:
    target.mkdir(parents=True, exist_ok=True)
```

`download_model` makes sure the model is present. It creates the folder.

**Lines 25-26**

```python
    if (target / "onnx" / "model.onnx").exists():
        return target
```

If the model file already exists, nothing is downloaded.

**Lines 27-29**

```python
    archive = target / "onnx.tar.gz"
    log(f"downloading {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, archive)
```

Otherwise it downloads the archive into the folder, printing a note first.

**Lines 30-33**

```python
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        archive.unlink()
        raise RuntimeError(f"model checksum mismatch: got {digest}, expected {MODEL_SHA256}")
```

Computes the downloaded file's SHA-256. If it is not the expected one, the file is **deleted** and an error is raised. The model is never used unless it matches.

**Lines 34-35**

```python
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")
```

Unpacks the archive. `filter="data"` is a safety setting that refuses unsafe paths inside the archive.

**Lines 36-38**

```python
    archive.unlink()
    log(f"model verified and extracted to {target}")
    return target
```

Deletes the archive, prints confirmation, and returns the folder.

### The MiniLMEmbedder

**Lines 41-42**

```python
class MiniLMEmbedder:
    kind = "onnx-minilm-l6-v2"
```

The class and its label (used to rebuild it later).

**Lines 44-47**

```python
    def __init__(self, model_dir: Path = DEFAULT_DIR):
        import onnxruntime as ort
        from tokenizers import Tokenizer
```

The constructor imports the model runtime (`onnxruntime`) and the text splitter (`tokenizers`) only when the class is used.

**Lines 48-50**

```python
        folder = model_dir / "onnx"
        if not (folder / "model.onnx").exists():
            raise FileNotFoundError(f"{folder}/model.onnx not found; run: python -m agentic_rag download-model")
```

Looks for the model file. If it is missing, it explains how to download it.

**Lines 51-52**

```python
        self.session = ort.InferenceSession(str(folder / "model.onnx"), providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
```

Loads the model for CPU use, and loads the tokenizer, which cuts text into the word-pieces the model understands.

**Lines 53-54**

```python
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
```

Tells the tokenizer to cut anything past 256 pieces, and to pad shorter texts with a filler piece so a batch of texts has equal length.

**Lines 55-56**

```python
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.dim = 384
```

Records which input names the model expects, and the output size, 384.

**Lines 58-59**

```python
    def fit(self, texts: list[str]) -> None:        # a pretrained model needs no fitting
        return None
```

`fit` does nothing: a pretrained model has nothing to learn from your documents.

**Lines 61-63**

```python
    def embed_many(self, texts: list[str], batch: int = 32) -> list[dict]:
        out: list[dict] = []
        for start in range(0, len(texts), batch):
```

`embed_many` embeds many texts quickly, 32 at a time (`batch`).

**Lines 64-66**

```python
            enc = self.tokenizer.encode_batch(texts[start:start + batch])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
```

For each batch: tokenizes the texts into numbers (`ids`), and a `mask` that marks real pieces versus filler.

**Lines 67-69**

```python
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
```

Builds the model's input. Some versions of the model also want `token_type_ids` (all zeros here).

**Line 70**

```python
            hidden = self.session.run(None, feed)[0]                       # (batch, tokens, 384)
```

Runs the model. The result is, for every piece of every text, a list of 384 numbers.

**Lines 71-72**

```python
            m = mask[:, :, None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)   # mean pooling over real tokens
```

**Mean pooling**: averages the 384-number lists of all real pieces of a text into a single list, ignoring filler pieces (the `mask` does this).

**Line 73**

```python
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
```

Scales each result to length 1, as the hash embedder does, so similarity is again a simple sum of products.

**Lines 74-75**

```python
            out += [{i: float(v) for i, v in enumerate(row)} for row in pooled]
        return out
```

Converts each result into the same dictionary format used everywhere else, and returns all of them.

**Lines 77-78**

```python
    def embed(self, text: str) -> dict:
        return self.embed_many([text])[0]
```

`embed` embeds one text by calling the batch method with a list of one.

**Lines 80-81**

```python
    def to_json(self) -> dict:
        return {"kind": self.kind, "dim": self.dim}
```

`to_json` returns the label and size to save in the database.

**Lines 83-85**

```python
    @classmethod
    def from_json(cls, data: dict) -> "MiniLMEmbedder":
        return cls()
```

`from_json` rebuilds the embedder (the model itself is loaded from disk, so only the label matters).


---

## `agentic_rag/retrieval.py`

Finding the chunks that hold the answer. Two complementary searches are run and merged: **meaning** (embeddings) catches paraphrases; **words** (BM25) catches exact terms, codes and numbers. A chunk ranked well by either rises. The file also contains *parent-child* matching: match on small pieces, return the bigger chunk.

### File description and imports

**Lines 1-7**

```python
"""Retrieval (plan section 6.3): find the chunks that hold the answer, then the sections they belong to.

    dense     meaning match: cosine similarity between the question and chunk embeddings
    bm25      word match: classic BM25 over the chunk text and its section title (exact terms, codes, numbers)
    hybrid    both, merged by reciprocal rank fusion (a chunk ranked well by either list rises)

Sections are ranked by their best chunk, so a fact buried deep in a section's body still lifts that section."""
```

The three modes (dense, bm25, hybrid) and the rule that sections are ranked by their best chunk, so a fact buried deep in a section still lifts that section.

**Lines 9-11**

```python
import math
import re
from collections import Counter, defaultdict
```

`math`, `re` and the `Counter`/`defaultdict` containers.

**Lines 13-15**

```python
from .embeddings import cosine
from .models import Chunk
from .text import content_stems
```

Imports the similarity function, the Chunk record and the stemming helper.

### BM25: word-based ranking

BM25 is a classic formula for ranking documents by the words of a query. Rare words count more, repetition counts with diminishing returns, and long documents are not unfairly favoured.

**Lines 18-20**

```python
class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
```

Takes the chunks as lists of stemmed words, and two tuning constants: `k1` (how fast repetition stops helping) and `b` (how much to correct for length).

**Lines 21-23**

```python
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(docs)) if docs else 1.0
```

Saves, for each chunk, how many times each word occurs (`tf`), its length, and the average length.

**Lines 24-26**

```python
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
```

Counts, for each word, how many chunks contain it (`df`).

**Lines 27-28**

```python
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
```

Gives each word an IDF weight, `log(1 + (n - df + 0.5)/(df + 0.5))`: a word in few chunks gets a high weight; a word in most chunks gets a weight near zero.

**Lines 30-32**

```python
    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, ln in zip(self.tf, self.len):
```

`scores` gives every chunk a score for a query. It loops over the chunks.

**Lines 33-35**

```python
            s = 0.0
            for t in set(query):
                f = tf.get(t, 0)
```

For each distinct query word: how often it appears in this chunk.

**Lines 36-37**

```python
                if f:
                    s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
```

If it appears, add `idf * f * (k1 + 1) / (f + k1 * (1 - b + b * length / average length))`. This is the BM25 formula: more occurrences raise the score, but with diminishing returns, and a long chunk is penalised a little.

**Lines 38-39**

```python
            out.append(s)
        return out
```

Stores this chunk's score, then returns all scores.

### The Retriever: dense, BM25 and hybrid

**Lines 42-45**

```python
class Retriever:
    def __init__(self, chunks: list[Chunk], embedder, mode: str = "hybrid", rrf_k: int = 60, dense_weight: float = 1.0):
        self.chunks, self.embedder, self.mode = chunks, embedder, mode
        self.rrf_k, self.dense_weight = rrf_k, dense_weight
```

Created from the chunks (which already carry their vectors), the embedder, the mode ('dense', 'bm25' or 'hybrid'), and two fusion settings.

**Line 46**

```python
        self.bm25 = BM25([content_stems(c.embed_text()) for c in chunks])
```

Builds the BM25 index from each chunk's embed text (prefix and text), stemmed.

**Lines 48-50**

```python
    def _dense(self, q: str) -> list[float]:
        qv = self.embedder.embed(q)
        return [cosine(qv, c.embedding) for c in self.chunks]
```

`_dense`: embeds the question, then scores every chunk by cosine similarity (meaning match).

**Lines 52-55**

```python
    @staticmethod
    def _ranks(scores: list[float]) -> dict[int, int]:
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return {i: r for r, i in enumerate(order, start=1)}
```

`_ranks`: turns scores into ranks. The best score gets rank 1, the next rank 2, and so on.

**Lines 57-60**

```python
    def rank_chunks(self, q: str, k: int = 10) -> list[tuple[Chunk, float]]:
        n = len(self.chunks)
        dense = self._dense(q) if self.mode in ("dense", "hybrid") else [0.0] * n
        sparse = self.bm25.scores(content_stems(q)) if self.mode in ("bm25", "hybrid") else [0.0] * n
```

`rank_chunks` returns the top `k` chunks. It computes the meaning scores and the word scores as the mode requires (unused ones are zeros).

**Lines 61-64**

```python
        if self.mode == "dense":
            fused = dense
        elif self.mode == "bm25":
            fused = sparse
```

In 'dense' mode the answer is the meaning score; in 'bm25' mode the word score.

**Lines 65-67**

```python
        else:
            rd, rs = self._ranks(dense), self._ranks(sparse)
            fused = [self.dense_weight / (self.rrf_k + rd[i]) + (1.0 / (self.rrf_k + rs[i]) if sparse[i] > 0 else 0.0) for i in range(n)]
```

In 'hybrid' mode the two rankings are merged by **reciprocal rank fusion**: each chunk earns `1/(60 + its meaning rank)` plus `1/(60 + its word rank)` (the word part only if it matched at all). A chunk near the top of both lists wins; a chunk near the top of just one still gets a good score. Using ranks instead of raw scores avoids mixing two different scales.

**Lines 68-69**

```python
        order = sorted(range(n), key=lambda i: fused[i], reverse=True)[:k]
        return [(self.chunks[i], fused[i]) for i in order]
```

Sorts by the merged score, keeps the top `k`, and returns (chunk, score) pairs.

**Lines 71-73**

```python
    def rank_chunks_detailed(self, q: str, k: int = 40) -> list[tuple[Chunk, float, float, int, int]]:
        """(chunk, fused score, dense cosine, dense rank, bm25 rank) best-first. The dense cosine is a meaningful absolute
        number with a neural embedder (relevant is high, unrelated is low); the two ranks show whether the rankers agree."""
```

`rank_chunks_detailed` does the hybrid merge but returns more for each chunk: the merged score, the raw **meaning similarity**, and both ranks. The pipeline uses the meaning similarity as an absolute 'how close is this?' number, and the two ranks to see whether the rankers agree.

**Lines 74-79**

```python
        n = len(self.chunks)
        dense, sparse = self._dense(q), self.bm25.scores(content_stems(q))
        rd, rs = self._ranks(dense), self._ranks(sparse)
        fused = [self.dense_weight / (self.rrf_k + rd[i]) + (1.0 / (self.rrf_k + rs[i]) if sparse[i] > 0 else 0.0) for i in range(n)]
        order = sorted(range(n), key=lambda i: fused[i], reverse=True)[:k]
        return [(self.chunks[i], fused[i], dense[i], rd[i], rs[i] if sparse[i] > 0 else n + 1) for i in order]
```

Computes both scores and ranks, merges them as above, and returns tuples. A chunk with no word match gets the worst possible word rank (`n + 1`).

**Lines 81-86**

```python
    @staticmethod
    def rank_sections(ranked: list[tuple[Chunk, float]]) -> list[tuple[str, float]]:
        best: dict[str, float] = {}
        for chunk, score in ranked:
            best.setdefault(chunk.branch_id, score)             # chunks arrive best-first, so the first one is the section's best
        return list(best.items())
```

`rank_sections` turns the ranked chunks into ranked sections. Because chunks arrive best first, the first chunk seen for a section is that section's best, and `setdefault` records it only once. The section order is the order of each section's best chunk.

### Parent-child matching (small to big)

A long table chunk holds many rows, so a question about one row looks only weakly similar to the whole chunk. Parent-child matching matches the question against single rows (or pairs of sentences), but hands back the whole parent chunk, so the answer still comes with its context.

**Lines 89-91**

```python
def make_units(chunk: Chunk, window: int = 2, include_whole: bool = True) -> list[str]:
    """Small units to MATCH on: each table row (with the table title and columns), or short sentence windows of a paragraph.
    A fact in one row of a long table, or in one sentence of a long paragraph, is no longer diluted by its neighbours."""
```

`make_units` splits one chunk into small units to match on. The docstring explains the point: a fact in one row is no longer diluted by its neighbours.

**Lines 92-93**

```python
    from .text import sentences
    prefix = chunk.context_prefix
```

Imports the sentence splitter and reads the chunk's prefix.

**Lines 94-96**

```python
    if "Rows:\n" in chunk.text:
        head, _, body = chunk.text.partition("Rows:\n")
        units = [f"{prefix} {head.strip()} {r[2:] if r.startswith('- ') else r}" for r in body.split("\n") if r.strip()]
```

If the chunk is a table (it contains 'Rows:'): split the header from the rows; each row becomes a unit consisting of the prefix, the table heading and that one row.

**Lines 97-99**

```python
    else:
        sents = sentences(chunk.text)
        units = [f"{prefix} {' '.join(sents[i:i + window])}" for i in range(0, max(1, len(sents) - window + 1))] if len(sents) > window else []
```

Otherwise it is prose: units are overlapping windows of 2 sentences, each with the prefix. A chunk of two or fewer sentences gets no windows.

**Lines 100-102**

```python
    if include_whole or not units:
        units.append(chunk.embed_text())
    return units
```

Also adds the whole chunk as one more unit (so broad matches still work). Returns the units.

**Lines 105-106**

```python
class SmallToBigRetriever:
    """Match on small units, return their parent chunks (plan 5.3, parent-child chunking). A parent is ranked by its best unit."""
```

`SmallToBigRetriever` wraps a normal `Retriever`. A parent chunk is ranked by its best-matching unit.

**Lines 108-110**

```python
    def __init__(self, chunks: list[Chunk], embedder, mode: str = "hybrid", window: int = 2, include_whole: bool = True,
                 dense_weight: float = 1.0, rrf_k: int = 60):
        self.parents = chunks
```

Takes the same inputs, plus the window size and whether to include whole chunks.

**Lines 111-115**

```python
        units, owner = [], []
        for i, c in enumerate(chunks):
            for text in make_units(c, window, include_whole):
                units.append(text)
                owner.append(i)
```

Makes all the units of all chunks, remembering which parent chunk each unit came from (`owner`).

**Line 116**

```python
        vecs = embedder.embed_many(units) if hasattr(embedder, "embed_many") else [embedder.embed(t) for t in units]
```

Embeds all the units (in batches if the embedder supports it).

**Lines 117-119**

```python
        self.unit_chunks = [Chunk(id=f"u{j}", branch_id=chunks[owner[j]].branch_id, text=units[j], context_prefix="", embedding=vecs[j])
                            for j in range(len(units))]
        self.owner = owner
```

Wraps each unit as a temporary chunk that keeps its parent's section id, and stores the owner list.

**Line 120**

```python
        self.inner = Retriever(self.unit_chunks, embedder, mode=mode, dense_weight=dense_weight, rrf_k=rrf_k)
```

Creates an ordinary Retriever over the units.

**Lines 122-133**

```python
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
```

`_parents` converts ranked units back to ranked parents. For each unit (best first) it finds the parent, skips parents already added, and stops when `k` parents are collected. The score information is passed along.

**Lines 135-136**

```python
    def rank_chunks(self, q: str, k: int = 10):
        return self._parents(self.inner.rank_chunks(q, k=k * 6), k)
```

`rank_chunks`: ranks units (asking for six times as many, since several units share a parent), then maps them to parents.

**Lines 138-139**

```python
    def rank_chunks_detailed(self, q: str, k: int = 40):
        return self._parents(self.inner.rank_chunks_detailed(q, k=k * 6), k)
```

`rank_chunks_detailed`: the same, keeping the extra information.

**Lines 141-143**

```python
    @staticmethod
    def rank_sections(ranked):
        return Retriever.rank_sections(ranked)
```

`rank_sections` is the same as the normal Retriever's.


---
