# Agentic RAG

A working implementation of the pipeline in [`Agentic_RAG_Pipeline_Plan.pdf`](Agentic_RAG_Pipeline_Plan.pdf):
hierarchical retrieval (**topic > type > text**) controlled by an agent loop that scores, decides, checks its own evidence,
retries, and says "I don't know" when it should.

**New to the code?** Read the line-by-line [code walkthrough](docs/walkthrough/README.md): every file explained section by section in plain English, in the order the data flows.

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
python3 -m unittest discover -s tests               # 60 unit tests (neural tests run once the model is downloaded)
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
| 5 (documents) PDF / DOCX / TXT loaders, tree, chunking, SQLite | `loaders.py`, `pdf_loader.py`, `structure.py`, `chunker.py`, `sqlite_store.py`, `ingest_docs.py` |
| 9 Evaluation: test set, per-type metrics | `evaluate.py`, `data/testset.jsonl`, `data/holdout.jsonl` |
| 13 Configuration reference | `config.py` |

## Ingesting PDF, Word and text files into SQLite

```bash
python3 -m agentic_rag ingest-docs "Rag testing doc.pdf" enterprise_company_policy_manual.docx enterprise_company_policy_manual.txt --reset
python3 -m agentic_rag db-info                        # what is stored, per document
python3 -m agentic_rag db-info --section 6.2          # the chunks of one subsection
python3 -m agentic_rag db-info --search "FIDO2 MFA"   # keyword search (SQLite FTS5)
```

Pipeline: **load** (`loaders.py`, `pdf_loader.py`) -> **structure** (`structure.py`: Section = topic, numbered subsection =
type) -> **chunk** (`chunker.py`) -> **SQLite** (`sqlite_store.py`: documents, sections, chunks, embeddings, FTS5 index)
-> **embeddings**. Re-running skips unchanged files (by SHA-256) and replaces changed ones.

What the chunker does: packs paragraphs to about 220 words on sentence boundaries; keeps each `MANDATORY CROSS-DOMAIN RULE`
paragraph as its own chunk tagged with its rule id; renders table rows as `Column: value | Column: value`, one row per
line, with the table title and column names repeated on every chunk; keeps charts as flagged figure chunks.
`SqliteStore.load_store()` rebuilds the in-memory store the `Pipeline` queries.

### Result on the three uploaded files (the same 30-page manual in three formats)

| | PDF | DOCX | TXT |
|---|---|---|---|
| Sections / subsections | 19 / 65 | 19 / 65 | 19 / 65 |
| Tables found | 30 | 30 | 30 |
| Chunks (paragraph / table / rule / figure) | 54 / 61 / 17 / 2 | 52 / 61 / 17 / 2 | 50 / 61 / 17 / 0 |

- Tables match **cell for cell** between PDF and DOCX (30 of 30) and between TXT and DOCX; paragraph text differs by
  about 0.01% of words.
- Chunk text is identical between formats for 89 to 97% of chunks.
- The five repeated golden rules are stored as 17 rule chunks: R-01 x4, R-02 x4, R-03 x3, R-04 x3, R-05 x3, matching the source.
- SQLite integrity check passes; every chunk has an embedding and a keyword-index row.

### Test questions for these documents

[`DOC_TEST_QUESTIONS.md`](DOC_TEST_QUESTIONS.md) lists 50 questions (machine-readable: `data/doc_questions.jsonl`): single facts, table
lookups, 12 multi-part questions that need more than one section, the five repeated golden rules, paraphrases, typos, a vague
question, follow-ups, and six that the documents cannot answer (including a dress-code trap and two chart-only questions).
`tests/test_doc_questions.py` checks the **answer key**: each expected fact must exist in the stored text of the PDF, the DOCX and
the TXT. These questions have not been run through the retrieval pipeline yet.

### Chunking, embeddings and retrieval

```bash
python3 -m agentic_rag download-model                                   # once: 83 MB MiniLM model, SHA-256 verified
python3 -m agentic_rag ingest-docs <files> --reset --embedder neural   # neural embeddings (use --embedder hash to skip the model)
python3 -m agentic_rag eval-docs --questions data/doc_questions_test.jsonl
```

What changed, measured **without the judgment layer** (does the retriever put the answer in front of the rest of the pipeline?).
65 answerable tuning questions and 22 answerable questions that were never tuned on, each asked of the PDF, DOCX and TXT:

| Retrieval setup | Section ranked 1st | Section in top 3 | Answer text in top 3 chunks | in top 5 chunks |
|---|---|---|---|---|
| Old: hash embeddings, similarity only (tuning) | 51% | 66% | 61% | 72% |
| **New: neural + BM25 hybrid (tuning)** | 66% | 88% | 83% | 91% |
| Old (untouched test set) | 42% | 64% | 58% | 64% |
| **New (untouched test set)** | **68%** | **91%** | **94%** | **100%** |

- **Embeddings:** `all-MiniLM-L6-v2` (neural, 384 dimensions, ONNX on CPU). It matches by meaning, so "how long are chat messages
  kept" finds "Ephemeral Messaging ... 180 calendar days". Inputs are cut at 256 word-pieces, so chunks stay under about 150 to 220 words.
- **Retrieval:** dense (meaning) and BM25 (words, with stemming) rankings merged by reciprocal rank fusion; sections are ranked by
  their best chunk. BM25 alone already beat the old setup, and the fusion beats either alone.
- **Chunking** (all tested; the first four changed nothing worth keeping): paragraph size 90 to 220 words, table group size,
  one-sentence overlap, and adding the section title to each chunk all landed within noise (about 2 points). **Row-level table chunks
  hurt** (answer text in top 3 fell from 83% to 74%) because answers often need several rows of one table. **Parent-child
  retrieval** (match on single rows and sentence pairs, return the whole parent chunk; `parent_child` in `config.py`) keeps the ranking
  and makes the similarity score sharper, which the "nothing matches" floor relies on.

End-to-end pass rate (judgment layer unchanged), before the retrieval work -> after:

| Question set | Before | After |
|---|---|---|
| 50 questions (used for tuning) | 56.7% | 60.0% |
| 25 questions (used for tuning) | 34.7% | 40.0% |
| **25 fresh questions (never tuned on)** | **40.0%** | **36.0%** |

**The retrieval is no longer the bottleneck; the judgment layer is.** On the fresh questions the answer text is in the top 5 chunks
for 100% of answerable questions, yet only 7 of 22 are answered correctly: 10 are refused or met with a clarifying question after
the right section was found (the keyword evidence check rejects paraphrases), 3 never reach the right section, 1 is answered from
the wrong section, 1 has the right section but the wrong facts. Better retrieval also makes the pipeline more willing to answer
weakly supported trap questions (for example "paying vendors in cryptocurrency", answered from the procurement section); the
relaxed evidence rule accepts them. Both are judgment-layer behaviours, left unchanged on purpose.
Samples are small (22 to 65 answerable questions): treat percentages as plus or minus 10 points.
Reports: `reports/docs_old_test.json`, `reports/docs_final_test.json`, `reports/docs_pc_*.json`.

### Known limits of this step

- **Charts are not searchable.** The PDF and DOCX contain 2 chart images whose numbers (SLA hours, control counts,
  retention years) appear nowhere else in their text. They are stored as flagged `figure` chunks with their captions
  (`needs_vision = 1`) and need OCR or a vision model to read. The TXT version includes written chart summaries.
- **Legacy `.doc` is not read.** Convert to `.docx` first (`soffice --headless --convert-to docx file.doc`). The uploaded
  "doc" was already a `.docx`.
- **Scanned PDFs are not supported** (no OCR). The PDF reader is built for PDFs with real text; it rebuilds tables from the
  thin rule lines this PDF draws, so a PDF with differently drawn tables may need extractor work.
- **Paragraph boundaries differ slightly in the PDF** (about 25 to 30 of ~155 paragraphs): bullet glyphs are not extractable,
  so a bullet list can merge into the heading above it, and the title banner is split differently.
- **The three files are one document.** Searching all three together returns each answer up to three times. Every chunk has a
  `content_hash` so duplicates can be collapsed in the next step.

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
