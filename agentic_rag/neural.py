"""A neural sentence-embedding model (all-MiniLM-L6-v2, ONNX) that understands meaning, not just shared words.

"How long are chat messages kept?" and "Ephemeral Messaging: 180 Calendar Days" share almost no words, but their embeddings are close.
Runs on CPU with onnxruntime. The model file (about 83 MB) is downloaded once by `python -m agentic_rag download-model`; its
SHA-256 is checked against the value published in ChromaDB's source, which distributes the same file.

Inputs longer than MAX_TOKENS word-pieces are cut off, so chunks should stay below roughly 150 words."""

import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

MODEL_URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
MODEL_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
DEFAULT_DIR = Path(os.environ.get("AGENTIC_RAG_MODEL_DIR", Path(__file__).resolve().parent.parent / "models" / "all-MiniLM-L6-v2"))
MAX_TOKENS = 256


def download_model(target: Path = DEFAULT_DIR, log=print) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    if (target / "onnx" / "model.onnx").exists():
        return target
    archive = target / "onnx.tar.gz"
    log(f"downloading {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        archive.unlink()
        raise RuntimeError(f"model checksum mismatch: got {digest}, expected {MODEL_SHA256}")
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")
    archive.unlink()
    log(f"model verified and extracted to {target}")
    return target


class MiniLMEmbedder:
    kind = "onnx-minilm-l6-v2"

    def __init__(self, model_dir: Path = DEFAULT_DIR):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        folder = model_dir / "onnx"
        if not (folder / "model.onnx").exists():
            raise FileNotFoundError(f"{folder}/model.onnx not found; run: python -m agentic_rag download-model")
        self.session = ort.InferenceSession(str(folder / "model.onnx"), providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.dim = 384

    def fit(self, texts: list[str]) -> None:        # a pretrained model needs no fitting
        return None

    def embed_many(self, texts: list[str], batch: int = 32) -> list[dict]:
        out: list[dict] = []
        for start in range(0, len(texts), batch):
            enc = self.tokenizer.encode_batch(texts[start:start + batch])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feed)[0]                       # (batch, tokens, 384)
            m = mask[:, :, None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)   # mean pooling over real tokens
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            out += [{i: float(v) for i, v in enumerate(row)} for row in pooled]
        return out

    def embed(self, text: str) -> dict:
        return self.embed_many([text])[0]

    def to_json(self) -> dict:
        return {"kind": self.kind, "dim": self.dim}

    @classmethod
    def from_json(cls, data: dict) -> "MiniLMEmbedder":
        return cls()
