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
    evidence_ratio_confident: float = 0.40   # lower bar when retrieval picked the branch with a clear lead
    sibling_margin: float = 0.15        # reject a branch if a sibling supports the question this much better
    # hybrid retrieval (used when the store holds neural embeddings)
    retrieval: str = "auto"            # auto = hybrid when embeddings are neural, else the older word-overlap ranking
    parent_child: bool = True           # match on small units (table rows, sentence pairs) and return their parent chunk
    dense_floor: float = 0.30           # best chunk's meaning-similarity below this = nothing in the knowledge base matches
    dense_gap_clear: float = 0.08       # lead in meaning-similarity that counts as a clear winner
    # agent loop and guardrails
    max_branch_tries: int = 3
    max_agent_steps: int = 12
    time_budget_seconds: float = 10.0
    # chunking
    chunk_size_words: int = 120
    chunks_to_llm: int = 6
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
    table_row_chunks: bool = False      # True: every table row is its own chunk (with the table title and column names)
    chunk_overlap_sentences: int = 0    # sentences repeated at the start of the next piece when a long paragraph is cut
