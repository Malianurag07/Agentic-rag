# Agentic RAG

A working implementation of the pipeline in [`Agentic_RAG_Pipeline_Plan.pdf`](Agentic_RAG_Pipeline_Plan.pdf):
hierarchical retrieval (**topic > type > text**) controlled by an agent loop that scores, decides, checks its own evidence,
retries, and says "I don't know" when it should.

Background reading: [`AGENTIC_RAG_EXPLAINED.md`](AGENTIC_RAG_EXPLAINED.md), [`RAG_AND_AGENTIC_RAG_DEEP_DIVE.md`](RAG_AND_AGENTIC_RAG_DEEP_DIVE.md).

## Quick start

Needs Python 3.11+. No packages are required for the default offline mode.

```bash
python3 -m agentic_rag ingest                       # build data/index.json from data/kb/*.md
python3 -m agentic_rag ask "I want money back for a broken item" --trace
python3 -m agentic_rag ask "What are the refund fraud checks?" --groups staff,public
python3 -m agentic_rag chat                         # follow-up questions work in a session
python3 -m agentic_rag eval                         # run data/testset.jsonl
python3 -m agentic_rag eval --testset data/holdout.jsonl --out reports/holdout.json
python3 -m unittest discover -s tests               # 20 unit tests
```

## How it maps to the plan

| Plan section | Code |
|---|---|
| 4 Data model (branches, chunks, metadata) | `agentic_rag/models.py` |
| 5 Ingestion: load, structure, chunk, enrich, embed, store | `agentic_rag/ingest.py`, `embeddings.py`, `store.py` |
| 6.1 Query prep (typos, follow-ups) | `HeuristicLLM.rewrite` in `llm.py` |
| 6.2 Permission filter, enforced in code | `Store.visible_*` in `store.py` |
| 6.3 Score branches (topic score + branch score) | `Pipeline._rank` in `pipeline.py` |
| 6.4 Score gap: clear winner skips the LLM, unclear asks it | `Pipeline._run_single`, `llm.choose` |
| 6.4 Topic-only question ("refund") asks the user | `Pipeline._topic_only` |
| 6.5 Fallback: search all chunks directly, then refuse honestly | `Pipeline._fallback` |
| 6.6 Evidence check with quotes per must-have fact, sibling comparison | `llm.check` |
| 6.7 Soft-block and retry (max 3), per question only | `Pipeline._run_single` |
| 6.8 Citations and faithfulness check | `Pipeline._respond`, `llm.unsupported` |
| 7 Tools: calculator, planner for "compare X and Y" | `tools.py`, `Pipeline._plan` |
| 8 Guardrails: step/time budget, loop detection, injection sanitizer, permissions, tracing | `Budget`, `tools.sanitize`, `Result.trace` |
| 9 Evaluation: test set, per-type metrics | `evaluate.py`, `data/testset.jsonl`, `data/holdout.jsonl` |
| 13 Configuration reference | `config.py` |

## Knowledge-base format

One Markdown file per topic in `data/kb/`. See the docstring in `agentic_rag/ingest.py`. Each type (`## Name`) can carry
`aliases:` (other ways users say it), a `description:`, and access/status metadata. Aliases matter a lot: they are how
"money back for a broken item" finds "Damage refund".

## Two modes

- **`--llm heuristic` (default).** Deterministic lexical rules for every judgment step, plus a local TF-IDF embedder.
  Runs anywhere with no API key. This is what the numbers below measure.
- **`--llm claude`.** Same interface, but Claude does the judgment steps (query rewrite, branch choice, quote-based
  evidence check, answer, faithfulness). Needs `pip install anthropic` and credentials. Models default to
  `claude-haiku-5-5` (routine steps) and `claude-opus-5-5` (judgment); override with `AGENTIC_RAG_SMALL_MODEL` /
  `AGENTIC_RAG_STRONG_MODEL`. **This mode has not been run against the live API yet**; it falls back to the heuristics if a
  reply can't be parsed.

## Results (heuristic mode, sample knowledge base: 5 topics, 15 active types)

| Test set | Questions | Pass rate | Branch top-1 | Branch top-3 |
|---|---|---|---|---|
| `data/testset.jsonl` (used while tuning) | 40 | 100% | 100% | 100% |
| `data/holdout.jsonl` (written afterwards, **not** tuned on) | 28 | **53.6%** | 45.5% | 90.9% |

**Read the second row, not the first.** The tuned set was fixed until it passed, so its 100% is optimistic and is only used
as a regression test. On fresh phrasing the offline mode fails mainly by refusing questions it could have answered:
the right branch is in the top 3 about 91% of the time, but the lexical evidence check rejects it when the question uses
words the knowledge base never contains ("vase", "faulty", "doorstep", "shoes"). It also lets through some
out-of-scope questions that share one known word ("Can I return a house?").

That is the limit of keyword rules, and it is the gap the plan assigns to an LLM judge and a neural embedding model. The
next steps, in order:

1. Run `--llm claude` on both sets and compare. That is the real test of the design.
2. Swap `HashEmbedder` for a neural embedding model (the interface is two methods: `fit`, `embed`).
3. Add aliases from real user questions. Failed questions in `reports/*.json` show where.
4. Grow the test set well beyond 68 questions before trusting any percentage.

## Not built yet (from the plan)

Response caching, multilingual embeddings, multi-parent branches (a chunk linked from two types), privacy masking in
traces, human-approval flow for write tools (version 1 is read-only), and the Phase 6 pilot with real users.
