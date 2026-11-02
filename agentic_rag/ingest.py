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

from pathlib import Path

from .config import Config
from .embeddings import HashEmbedder
from .models import Branch, Chunk
from .store import Store
from .text import sentences, slug

META_KEYS = ("aliases", "description", "access", "status")


def _split_list(value: str, sep: str = ";") -> list[str]:
    return [v.strip() for v in value.split(sep) if v.strip()]


def parse_document(path: Path) -> tuple[dict, list[dict]]:
    """Return (front_matter, sections). Each section: {name, meta, text}."""
    raw = path.read_text().splitlines()
    if not raw or raw[0].strip() != "---":
        raise ValueError(f"{path}: missing front matter")
    end = raw[1:].index("---") + 1
    front = {}
    for line in raw[1:end]:
        if ":" in line:
            k, v = line.split(":", 1)
            front[k.strip().lower()] = v.split("  #")[0].strip()
    sections: list[dict] = []
    current = None
    for line in raw[end + 1:]:
        if line.startswith("## "):
            current = {"name": line[3:].strip(), "meta": {}, "lines": []}
            sections.append(current)
        elif current is not None:
            key = line.split(":", 1)[0].strip().lower()
            if ":" in line and key in META_KEYS:
                current["meta"][key] = line.split(":", 1)[1].strip()
            else:
                current["lines"].append(line)
    for s in sections:
        s["text"] = "\n".join(s.pop("lines")).strip()
    return front, sections


def chunk_text(text: str, max_words: int) -> list[str]:
    """Pack paragraphs, then sentences, into chunks of roughly max_words, without cutting a sentence in half."""
    pieces: list[str] = []
    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        pieces.extend(sentences(para) if len(para.split()) > max_words else [para])
    chunks, buf, count = [], [], 0
    for piece in pieces:
        n = len(piece.split())
        if buf and count + n > max_words:
            chunks.append(" ".join(buf))
            buf, count = [], 0
        buf.append(piece)
        count += n
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def build_store(kb_dir: str | Path, config: Config | None = None) -> Store:
    config = config or Config()
    branches: dict[str, Branch] = {}
    chunks: list[Chunk] = []

    for path in sorted(Path(kb_dir).glob("*.md")):
        front, sections = parse_document(path)
        topic_name = front["topic"]
        access = _split_list(front.get("access", "public"), ",")
        status = front.get("status", "active")
        version = front.get("version", "1")
        source = front.get("source", path.stem)
        topic_id = slug(topic_name)

        if topic_id not in branches:
            branches[topic_id] = Branch(
                id=topic_id, name=topic_name, level=1, parent_id=None, path=topic_name,
                description=front.get("description", ""), aliases=_split_list(front.get("aliases", "")),
                access=access, status="active")
        else:  # a second file for the same topic only adds to its aliases and widens access
            existing = branches[topic_id]
            existing.aliases += [a for a in _split_list(front.get("aliases", "")) if a not in existing.aliases]
            existing.access = sorted(set(existing.access) | set(access))

        for sec in sections:
            suffix = f"@{version}" if status != "active" else ""
            bid = f"{topic_id}.{slug(sec['name'])}{suffix}"
            meta = sec["meta"]
            branch = Branch(
                id=bid, name=sec["name"], level=2, parent_id=topic_id, path=f"{topic_name} > {sec['name']}",
                description=meta.get("description", ""), aliases=_split_list(meta.get("aliases", "")),
                access=_split_list(meta["access"], ",") if "access" in meta else access,
                status=meta.get("status", status), version=version)
            branches[bid] = branch
            for i, piece in enumerate(chunk_text(sec["text"], config.chunk_size_words), 1):
                chunks.append(Chunk(
                    id=f"{bid}.{i:03d}", branch_id=bid, text=piece, context_prefix=f"{branch.path}:",
                    source=source, access=branch.access, status=branch.status))

    embedder = HashEmbedder(dim=config.embedding_dim)
    embedder.fit([b.search_text() for b in branches.values()] + [c.embed_text() for c in chunks])
    for b in branches.values():
        b.embedding = embedder.embed(b.search_text())
    for c in chunks:
        c.embedding = embedder.embed(c.embed_text())
    return Store(embedder, list(branches.values()), chunks)
