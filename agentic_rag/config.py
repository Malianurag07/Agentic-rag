"""Every tunable value lives here (plan section 13). The defaults are starting points, not proven constants:
tune them with `python -m agentic_rag eval` on your own test set."""

from dataclasses import dataclass, field


@dataclass
class Config:
    # retrieval
    top_k_branches: int = 10
    topic_weight: float = 0.35          # weight of the topic score in a leaf's combined score
    score_floor: float = 0.12           # best combined score below this means "nothing matches"
    score_gap_clear: float = 0.10       # best minus second best at or above this means "clear winner"
    llm_candidates_shown: int = 5
    ask_options_gap: float = 0.15       # options offered in a clarifying question must be this close to the best
    # evidence check
    evidence_ratio: float = 0.60        # share of (weighted) question facts that must be supported
    unknown_term_weight: float = 1.00   # an unknown word counts this share of the heaviest known word
    sibling_margin: float = 0.15        # reject a branch if a sibling supports the question this much better
    # agent loop and guardrails
    max_branch_tries: int = 3
    max_agent_steps: int = 12
    time_budget_seconds: float = 10.0
    # chunking
    chunk_size_words: int = 120
    chunks_to_llm: int = 4
    # fallback search
    fallback_top_k: int = 5
    fallback_floor: float = 0.08
    keyword_weight: float = 0.30
    # embeddings
    embedding_dim: int = 2048
    # models used by the optional Claude backend
    small_model: str = "claude-haiku-5-5"
    strong_model: str = "claude-opus-5-5"
    public_groups: tuple = field(default=("public",))
    # document chunking (policy manuals, PDFs, Word files)
    doc_chunk_words: int = 220          # paragraphs are packed up to about this many words
    table_chunk_words: int = 160        # table rows are packed up to about this many words (header repeated on each chunk)
