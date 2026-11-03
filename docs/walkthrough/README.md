# Code walkthrough: how the Agentic RAG pipeline works, section by section

This is a guided tour of the pipeline's source code, written in plain English. Each file is split into sections (imports, constants, one section per function or idea), and **every line of code is shown next to an explanation of what it does**. Read it like a book, in order.

- It covers the 18 core files, about 2,380 lines of code, in the order the data flows through the system.
- The code shown is **copied from the real files by line number**, never retyped. It matches commit `9991942` (`Neural embeddings, hybrid retrieval and parent-child chunk matching`). If the code changes later, the line numbers here will drift.
- **Every non-blank line of those files is explained.** A script checks this: the build fails if any line is left out. A second script checks that the identifiers named in each explanation appear next to the code they describe (716 explanations checked).
- The command-line wrapper (`cli.py`), the evaluation harnesses (`evaluate.py`, `doc_eval.py`, `retrieval_eval.py`) and the tests are not covered here. They measure and drive the pipeline but are not part of it.

## The big picture

The system has two halves that share one database.

**Half 1: ingestion (done once per document)**

```
file (PDF, Word or text)
   |  loaders.py / pdf_loader.py      read it into Blocks (paragraphs, tables, figures)
   v
Blocks
   |  structure.py                    find "Section 3" / "3.2 Title" headings -> a tree
   v
Tree of sections and subsections
   |  chunker.py                      cut each subsection into searchable chunks
   v
Chunks
   |  sqlite_store.py                 store text + keyword index in SQLite
   |  embeddings.py / neural.py       turn every chunk into numbers (meaning)
   v
SQLite database (documents, sections, chunks, vectors, keyword index)
```

**Half 2: answering a question (every time someone asks)**

```
question
   |  pipeline.py: split multi-part questions, rewrite typos and follow-ups (llm.py)
   v
retrieval.py: find the best chunks two ways (by meaning and by words), merge the rankings
   |
   v
pipeline.py: pick a section   -- confident? take it. unsure? ask the judge. vague? ask the user.
   |
   v
llm.py: evidence check        -- does this text really answer the question?
   |        no  -> reject it, try the next candidate (up to 3), then fall back / refuse
   |        yes
   v
llm.py: write the answer, check each sentence is supported; tools.py: calculator, hostile-text filter
   |
   v
answer + citation   (or an honest "I couldn't find a reliable answer")
```

## Reading order

| Part | File(s) | What you will learn |
|---|---|---|
| [1. Foundation](01_foundation.md) | `blocks.py`, `config.py`, `models.py`, `text.py` | The data shapes, every setting, and the tiny text tools everything shares |
| [2. Loading documents](02_loading.md) | `loaders.py`, `pdf_loader.py` | How text, Word and PDF files become Blocks. The PDF reader rebuilds tables, columns and paragraphs from raw coordinates |
| [3. Structure and chunking](03_structure_and_chunking.md) | `structure.py`, `chunker.py` | How headings are detected, and how a section is cut into chunks |
| [4. Storage and ingestion](04_storage_and_ingestion.md) | `sqlite_store.py`, `ingest_docs.py`, `store.py`, `ingest.py` | The database, the in-memory store with permission filtering, and the ingestion drivers |
| [5. Embeddings and retrieval](05_embeddings_and_retrieval.md) | `embeddings.py`, `neural.py`, `retrieval.py` | How text becomes numbers, how the neural model is loaded safely, and how BM25 and meaning-search are merged |
| [6. Tools and the judgment layer](06_tools_and_judgment.md) | `tools.py`, `llm.py` | The safe calculator, the hostile-text filter, and the judge that checks evidence |
| [7. The agent pipeline](07_the_agent_pipeline.md) | `pipeline.py` | The loop that answers a question, with its safety limits |

## Words you will meet

| Word | Plain meaning |
|---|---|
| **Block** | One piece of a document straight after loading: a paragraph, a table or a figure |
| **Chunk** | A piece of text small enough to be found and shown on its own. The unit of search |
| **Branch / section** | A node in the table of contents: a topic (Section 3) or a type (3.2). Chunks sit under subsections |
| **Embedding** | A list of numbers that represents the meaning of a text. Similar meanings give similar numbers |
| **Cosine similarity** | A score from 0 to 1 for how alike two embeddings are |
| **BM25** | A classic formula that ranks text by the words in the question. Rare words count more |
| **Hybrid retrieval** | Run both searches (meaning and words) and merge the two rankings |
| **Reciprocal rank fusion** | The merge: each chunk earns `1/(60 + its rank)` from each list and the points are added |
| **Parent-child matching** | Match the question against small pieces (a table row), but return the bigger chunk it belongs to |
| **Evidence check** | The step that asks: is every key fact of the question supported by this text? |
| **Soft-block** | Marking a rejected section as rejected *for this question only*, without deleting it |
| **Fallback** | The safety net: a search over everything when the normal route fails. If that fails too, the system refuses honestly |
| **Prompt injection** | Hostile text hidden in a document that tries to give orders to the AI. The pipeline deletes such sentences |
| **Judge (HeuristicLLM / ClaudeLLM)** | The component that makes the five judgment calls. The default one uses word-matching rules; the other asks Claude |

## Things noticed while writing this (not changed)

These are small oddities in the code, listed so you are not surprised by them. None is a bug that breaks results.

1. **`pipeline.py` line 150:** `(cands[0].score if self.hybrid else cands[0].score)` gives the same value in both branches, so the `if` does nothing.
2. **`pipeline.py` line 209:** the comment about ranking sits above `_unknown_terms`, which is not a ranking function. It belongs above the ranking functions further down.
3. **`pipeline.py` line 201:** when retrying the next candidate, the cut-off uses the older similarity floor (0.12), even in hybrid mode where the main floor is 0.30. The two numbers measure the same thing, so this mixes two standards.
4. **`pipeline.py` line 334:** the fallback raises `LoopDetected` just to jump to the refusal at the bottom. It works, but the name is misleading.
5. **Unused imports:** `re` and `defaultdict` in `retrieval.py`, `HashEmbedder` in `sqlite_store.py`, `Block` in `chunker.py`.
6. **`ClaudeLLM`** has never been run against the live service in this project (no API key was available). It is written to fall back to the rule-based judge on any error.
