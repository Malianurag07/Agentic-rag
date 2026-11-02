"""The two record types everything rests on (plan section 4): branches (the tree) and chunks (the text)."""

from dataclasses import dataclass, field


@dataclass
class Branch:
    id: str
    name: str
    level: int                      # 1 = topic, 2 = type
    parent_id: str | None
    path: str                       # "Refund > Damage refund"
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    access: list[str] = field(default_factory=lambda: ["public"])
    status: str = "active"
    version: str = "1"
    effective_date: str = ""
    embedding: dict = field(default_factory=dict)

    def search_text(self) -> str:
        return " ".join([self.path, self.description, " ".join(self.aliases)])


@dataclass
class Chunk:
    id: str
    branch_id: str
    text: str
    context_prefix: str
    source: str = ""
    access: list[str] = field(default_factory=lambda: ["public"])
    status: str = "active"
    embedding: dict = field(default_factory=dict)

    def embed_text(self) -> str:
        return f"{self.context_prefix} {self.text}"


@dataclass
class Candidate:
    branch: Branch
    score: float
    topic_score: float = 0.0
    child_score: float = 0.0
    blocked_reason: str | None = None    # set when soft-blocked for the current question


@dataclass
class Choice:
    branch_id: str | None
    reason: str = ""
    ask: str | None = None               # a clarifying question for the user, when no branch fits clearly


@dataclass
class Check:
    ok: bool
    reason: str
    supports: list = field(default_factory=list)   # (must-have fact, quote) pairs
    missing: list = field(default_factory=list)
    ratio: float = 0.0
    unknown: list = field(default_factory=list)   # question words that appear nowhere in the knowledge base
