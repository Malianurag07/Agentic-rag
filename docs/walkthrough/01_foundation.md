# Part 1: The foundation (settings, data shapes, text helpers)

These four small files are used by almost every other file. Read them first: they define the **words** the rest of the code speaks in. `config.py` holds every setting, `blocks.py` and `models.py` define the shapes of the data that flows through the pipeline, and `text.py` holds the tiny text tools (splitting words, cutting endings off words) that everything shares.

## `agentic_rag/blocks.py`

A **Block** is the first thing every document becomes. Whether the file is a PDF, a Word file or a text file, the loader turns it into a plain list of Blocks (paragraphs, tables, figures). After that point the rest of the code does not care what the original format was.

### File description

**Lines 1-2**

```python
"""The format-neutral intermediate form. Every loader (TXT, DOCX, PDF) produces a list of Blocks, so structure detection,
chunking and storage are written once and behave the same for all formats."""
```

A docstring: a note that describes the file. It says every loader (text, Word, PDF) produces Blocks, so the later steps (finding headings, chunking, storing) are written once and work the same for all formats.

### Imports

**Line 4**

```python
from dataclasses import dataclass, field
```

Brings in two tools from Python's standard library. `dataclass` turns a plain class into a simple record: Python writes the setup code for us. `field` lets us give a field a safe default such as an empty list (a plain `[]` default would be shared between all records, which is a classic bug).

### The Block record

**Lines 7-8**

```python
@dataclass
class Block:
```

`@dataclass` marks the class below it as a record. `class Block:` defines it. One Block is one piece of a document.

**Line 9**

```python
    kind: str                         # "para" | "table" | "figure"
```

`kind` says what sort of piece it is: `"para"` (a paragraph), `"table"` or `"figure"` (an image). It has no default, so it must always be given.

**Line 10**

```python
    text: str = ""                    # paragraph text, or a caption/placeholder for a figure
```

`text` holds a paragraph's words, or for a figure a caption or placeholder note. It defaults to an empty string because tables do not use it.

**Line 11**

```python
    header: list = field(default_factory=list)    # table column titles
```

`header` holds a table's column titles, as a list. `default_factory=list` means each Block gets its own fresh empty list.

**Line 12**

```python
    rows: list = field(default_factory=list)      # table body rows (lists of cell strings)
```

`rows` holds the body of a table: a list of rows, each row being a list of cell texts.

**Line 13**

```python
    page: int | None = None           # PDF page number (1-based); None for formats without pages
```

`page` is the PDF page number the block came from (starting at 1). The type `int | None` means it is a whole number or nothing. Word and text files have no pages, so it stays `None`.

**Line 14**

```python
    meta: dict = field(default_factory=dict)
```

`meta` is a free notebook (a dictionary) for extra facts, for example the size of an image or the last page of a table that continued across pages.

### Counting words

**Line 16**

```python
    def words(self) -> int:
```

A small method that answers: how many words does this block contain?

**Lines 17-18**

```python
        if self.kind == "table":
            return sum(len(c.split()) for r in [self.header] + self.rows for c in r)
```

If the block is a table, it loops over the header row plus every body row, and inside each row over every cell, splitting each cell into words and adding up the counts. The double loop in one line is a *generator expression*.

**Line 19**

```python
        return len(self.text.split())
```

Otherwise (paragraph or figure) it simply splits the text on spaces and counts the pieces.


---

## `agentic_rag/config.py`

Every number you might want to change lives in one place: the `Config` class. The rest of the code never hard-codes a threshold; it reads it from here. That makes tuning easy, and you can see every knob at a glance. The values are *starting points*, not proven constants.

### File description and imports

**Lines 1-2**

```python
"""Every tunable value lives here (plan section 13). The defaults are starting points, not proven constants:
tune them with `python -m agentic_rag eval` on your own test set."""
```

Says what the file is for: all tunable values, collected in one place. The values are starting points that should be tuned by running the evaluation on your own questions.

**Line 4**

```python
from dataclasses import dataclass, field
```

Same two tools as before: `dataclass` makes a record, `field` allows safe defaults for lists and tuples.

### Class and the retrieval settings

**Lines 7-8**

```python
@dataclass
class Config:
```

`Config` is a record where every field has a default. `Config()` gives the defaults; `Config(max_agent_steps=5)` changes just one.

**Line 9**

```python
    # retrieval
```

A comment that labels the group of settings below it: how the system looks for the right section.

**Line 10**

```python
    top_k_branches: int = 10
```

`top_k_branches`: how many candidate sections to keep after ranking (10).

**Line 11**

```python
    topic_weight: float = 0.35          # weight of the topic score in a leaf's combined score
```

`topic_weight`: used by the older ranking method. A section's score blends its own score with its parent topic's score; this is the weight of the parent's part (35%).

**Line 12**

```python
    score_floor: float = 0.12           # best combined score below this means "nothing matches"
```

`score_floor`: in the older method, if even the best section scores below 0.12, the system concludes nothing in the knowledge base matches and refuses.

**Line 13**

```python
    score_gap_clear: float = 0.10       # best minus second best at or above this means "clear winner"
```

`score_gap_clear`: in the older method, if the best section beats the second best by at least 0.10, it is a clear winner and the system skips the extra 'which one do you mean' step.

**Line 14**

```python
    llm_candidates_shown: int = 5
```

`llm_candidates_shown`: when the system does ask the judge to pick, it shows the top 5 candidates.

**Line 15**

```python
    ask_options_gap: float = 0.15       # options offered in a clarifying question must be this close to the best
```

`ask_options_gap`: when the system asks the user a clarifying question, it only offers options whose score is within 0.15 of the best.

### Evidence check settings

**Line 16**

```python
    # evidence check
```

Label: the settings for the step that checks whether the found text really supports an answer.

**Line 17**

```python
    evidence_ratio: float = 0.60        # share of (weighted) question facts that must be supported
```

`evidence_ratio`: the share of the question's key facts that must be supported by the text. 0.60 means 60%, weighted so rare words count more.

**Line 18**

```python
    unknown_term_weight: float = 1.00   # an unknown word counts this share of the heaviest known word
```

`unknown_term_weight`: how much a question word that appears *nowhere* in the knowledge base counts against the answer. 1.0 means as much as the heaviest known word, so a word like 'car' in a policy manual sinks the question.

**Line 19**

```python
    evidence_ratio_confident: float = 0.40   # lower bar when retrieval picked the branch with a clear lead
```

`evidence_ratio_confident`: a lower bar (40%) used when retrieval had a clear lead on the chosen section.

**Line 20**

```python
    sibling_margin: float = 0.15        # reject a branch if a sibling supports the question this much better
```

`sibling_margin`: if a neighbouring section supports the question at least 0.15 better than the chosen one, the chosen one is rejected.

### Hybrid retrieval settings

**Line 21**

```python
    # hybrid retrieval (used when the store holds neural embeddings)
```

Label: settings for the newer retrieval method that combines meaning and word matching.

**Line 22**

```python
    retrieval: str = "auto"            # auto = hybrid when embeddings are neural, else the older word-overlap ranking
```

`retrieval`: `"auto"` means use the hybrid method when the stored embeddings are neural, otherwise fall back to the older word-overlap ranking.

**Line 23**

```python
    parent_child: bool = True           # match on small units (table rows, sentence pairs) and return their parent chunk
```

`parent_child`: when true, the system matches the question against small pieces (single table rows, pairs of sentences) but hands back the whole larger chunk they came from.

**Line 24**

```python
    dense_floor: float = 0.30           # best chunk's meaning-similarity below this = nothing in the knowledge base matches
```

`dense_floor`: with neural embeddings, if the best chunk's *meaning similarity* is under 0.30, nothing in the knowledge base is close enough, so refuse.

**Line 25**

```python
    dense_gap_clear: float = 0.08       # lead in meaning-similarity that counts as a clear winner
```

`dense_gap_clear`: a lead of 0.08 in meaning similarity counts as a clear winner.

### Agent loop and guardrails

**Line 26**

```python
    # agent loop and guardrails
```

Label: the safety limits that stop the agent running forever.

**Line 27**

```python
    max_branch_tries: int = 3
```

`max_branch_tries`: if the evidence check rejects a section, move to the next one, but at most 3 times.

**Line 28**

```python
    max_agent_steps: int = 12
```

`max_agent_steps`: the total number of actions (searches, checks, answers) one question may use: 12.

**Line 29**

```python
    time_budget_seconds: float = 10.0
```

`time_budget_seconds`: one question may use at most 10 seconds.

### Chunking, fallback search, embeddings, models

**Line 30**

```python
    # chunking
```

Label: chunking settings for the small sample knowledge base.

**Line 31**

```python
    chunk_size_words: int = 120
```

`chunk_size_words`: target chunk size (120 words) for the sample knowledge base.

**Line 32**

```python
    chunks_to_llm: int = 6
```

`chunks_to_llm`: how many chunks of a chosen section are handed to the judge and answer writer (6).

**Line 33**

```python
    # fallback search
```

Label: the fallback search, used when the main route fails.

**Line 34**

```python
    fallback_top_k: int = 5
```

`fallback_top_k`: the fallback looks at the top 5 chunks.

**Line 35**

```python
    fallback_floor: float = 0.08
```

`fallback_floor`: in the older fallback, a chunk must score at least 0.08 to count.

**Line 36**

```python
    keyword_weight: float = 0.30
```

`keyword_weight`: in the older methods, how much shared words add to a similarity score (0.30).

**Line 37**

```python
    # embeddings
```

Label: embeddings.

**Line 38**

```python
    embedding_dim: int = 2048
```

`embedding_dim`: the size of the older hashed vectors (2048 slots). The neural model has its own fixed size of 384.

**Line 39**

```python
    # models used by the optional Claude backend
```

Label: the optional Claude backend.

**Lines 40-41**

```python
    small_model: str = "claude-haiku-5-5"
    strong_model: str = "claude-opus-5-5"
```

The names of the two Claude models to use: a small cheap one for routine steps and a strong one for judgment.

**Line 42**

```python
    public_groups: tuple = field(default=("public",))
```

`public_groups`: the default access group, `"public"`. A *tuple* (fixed list) is used, written with `field(default=...)`.

### Document chunking settings

**Line 43**

```python
    # document chunking (policy manuals, PDFs, Word files)
```

Label: chunking for real documents (policy manuals, PDFs, Word files).

**Line 44**

```python
    doc_chunk_words: int = 220          # paragraphs are packed up to about this many words
```

`doc_chunk_words`: paragraphs are packed together up to about 220 words per chunk.

**Line 45**

```python
    table_chunk_words: int = 160        # table rows are packed up to about this many words (header repeated on each chunk)
```

`table_chunk_words`: table rows are packed up to about 160 words per chunk; the table title and column names are repeated on every chunk.

**Line 46**

```python
    table_row_chunks: bool = False      # True: every table row is its own chunk (with the table title and column names)
```

`table_row_chunks`: if true, every table row becomes its own chunk. Off by default because testing showed it made answers worse (answers often need several rows).

**Line 47**

```python
    chunk_overlap_sentences: int = 0    # sentences repeated at the start of the next piece when a long paragraph is cut
```

`chunk_overlap_sentences`: how many sentences to repeat at the start of the next piece when a long paragraph has to be cut. 0 means no overlap.


---

## `agentic_rag/models.py`

The five record types that move through the pipeline. **Branch** and **Chunk** are what is stored; **Candidate**, **Choice** and **Check** are what the agent produces while thinking about one question. Think of them as the forms the staff fill in: a Branch is a shelf label, a Chunk is a page on the shelf, a Candidate is 'a shelf we might look at', a Choice is 'the shelf the judge picked', and a Check is 'the verdict on whether the page answers the question'.

### File description and imports

**Line 1**

```python
"""The two record types everything rests on (plan section 4): branches (the tree) and chunks (the text)."""
```

Says these are the two record types everything rests on: branches (the tree of topics) and chunks (the text).

**Line 3**

```python
from dataclasses import dataclass, field
```

Imports `dataclass` and `field`, as before.

### Branch: one node of the topic tree

**Lines 6-7**

```python
@dataclass
class Branch:
```

A Branch is one entry in the table of contents: either a topic (like 'Refund') or a type under it (like 'Damage refund'). In the policy manuals a topic is a numbered section and a type is a numbered subsection.

**Lines 8-9**

```python
    id: str
    name: str
```

`id` is a unique code for the branch; `name` is its human title.

**Line 10**

```python
    level: int                      # 1 = topic, 2 = type
```

`level`: 1 for a topic, 2 for a type.

**Line 11**

```python
    parent_id: str | None
```

`parent_id`: the id of the topic above it, or `None` for a topic (topics have no parent).

**Line 12**

```python
    path: str                       # "Refund > Damage refund"
```

`path`: the full breadcrumb, for example 'Refund > Damage refund'. It is shown in citations.

**Lines 13-14**

```python
    description: str = ""
    aliases: list[str] = field(default_factory=list)
```

`description` is a short summary, and `aliases` is a list of other ways users say the same thing (for example 'broken item'). Both help matching.

**Line 15**

```python
    access: list[str] = field(default_factory=lambda: ["public"])
```

`access`: which groups of users may see this branch. Defaults to `["public"]`. The `lambda` makes a new list for each branch.

**Lines 16-18**

```python
    status: str = "active"
    version: str = "1"
    effective_date: str = ""
```

`status` ('active' or 'retired'), `version` and `effective_date` support old-versus-new policy handling: retired branches are never shown.

**Line 19**

```python
    embedding: dict = field(default_factory=dict)
```

`embedding`: the branch's meaning-vector (a dictionary from position to number), filled in after ingestion.

**Lines 21-22**

```python
    def search_text(self) -> str:
        return " ".join([self.path, self.description, " ".join(self.aliases)])
```

`search_text()` glues together the path, description and aliases into one string. That string is what gets turned into the branch's embedding.

### Chunk: one piece of searchable text

**Lines 25-26**

```python
@dataclass
class Chunk:
```

A Chunk is a small piece of text that can be found and shown. It is the unit of retrieval.

**Lines 27-29**

```python
    id: str
    branch_id: str
    text: str
```

`id`, the `branch_id` it belongs to (which section it sits under), and the `text` itself.

**Line 30**

```python
    context_prefix: str
```

`context_prefix`: a short label such as '4.3 Records Retention:' that is placed in front of the text when it is embedded, so the chunk never loses track of where it came from.

**Line 31**

```python
    source: str = ""
```

`source`: the file or document name, used in citations.

**Lines 32-34**

```python
    access: list[str] = field(default_factory=lambda: ["public"])
    status: str = "active"
    embedding: dict = field(default_factory=dict)
```

`access`, `status` and `embedding`: the same ideas as for branches.

**Lines 36-37**

```python
    def embed_text(self) -> str:
        return f"{self.context_prefix} {self.text}"
```

`embed_text()` returns the prefix plus the text. This combined string is what the embedder reads, and what the keyword search indexes.

### Candidate: a section the agent is considering

**Lines 40-42**

```python
@dataclass
class Candidate:
    branch: Branch
```

A Candidate wraps a Branch together with how well it matches the current question.

**Line 43**

```python
    score: float
```

`score`: the match score. (With hybrid retrieval it holds the best meaning-similarity among the section's chunks.)

**Lines 44-45**

```python
    topic_score: float = 0.0
    child_score: float = 0.0
```

`topic_score` and `child_score`: the two parts of the older combined score (parent topic part and the section's own part).

**Line 46**

```python
    blocked_reason: str | None = None    # set when soft-blocked for the current question
```

`blocked_reason`: filled in if the evidence check rejected this section for the current question (a 'soft block': it stays in the list, just marked).

**Line 47**

```python
    agree: bool = False                  # hybrid retrieval: the dense and keyword rankers both put this section first (or it leads clearly)
```

`agree`: with hybrid retrieval, true when both rankers (meaning and keywords) put this section first, or it leads clearly. It is the pipeline's confidence signal.

### Choice and Check: the judge's outputs

**Lines 50-52**

```python
@dataclass
class Choice:
    branch_id: str | None
```

A Choice is the result of the step 'pick the best section from these candidates'. `branch_id` is the section picked, or `None` if no section was picked.

**Line 53**

```python
    reason: str = ""
```

`reason`: a short note on why that section was picked.

**Line 54**

```python
    ask: str | None = None               # a clarifying question for the user, when no branch fits clearly
```

`ask`: if the question was too vague to choose, this holds a question to put back to the user, such as 'Which refund do you mean?'

**Lines 57-59**

```python
@dataclass
class Check:
    ok: bool
```

A Check is the verdict of the evidence check on one section. `ok` is true if the text supports an answer.

**Line 60**

```python
    reason: str
```

`reason`: a plain sentence saying why the check passed or failed.

**Lines 61-62**

```python
    supports: list = field(default_factory=list)   # (must-have fact, quote) pairs
    missing: list = field(default_factory=list)
```

`supports` holds pairs of (a fact from the question, the sentence that proves it). `missing` lists the question facts the text did not support.

**Line 63**

```python
    ratio: float = 0.0
```

`ratio`: the share of the question's key facts that were supported, from 0 to 1.

**Line 64**

```python
    unknown: list = field(default_factory=list)   # question words that appear nowhere in the knowledge base
```

`unknown`: question words that appear nowhere in the whole knowledge base, such as 'car' in a policy manual. They are a strong signal that the question is outside the documents.


---

## `agentic_rag/text.py`

Tiny text tools used by every stage. They are deliberately simple. The most important idea: the **same** functions process both the documents and the questions, so a word is cut to the same stem on both sides and can be matched. For example 'refunded', 'refunds' and 'refund' all become `refund`.

### File description, import and patterns

**Line 1**

```python
"""Small text helpers shared by every stage (tokenising, light stemming, sentence splitting)."""
```

Describes the file: small text helpers (splitting into words, light stemming, splitting into sentences).

**Line 3**

```python
import re
```

Imports `re`, Python's regular-expression tool, for pattern matching in text.

**Line 5**

```python
_WORD = re.compile(r"[a-z0-9$%]+(?:'[a-z]+)?")
```

`_WORD` is a pattern that finds words: letters, digits, `$` and `%` (so '$5,000' and '15%' survive as tokens), optionally followed by an apostrophe part like "n't". The leading underscore signals 'private to this file'.

**Line 6**

```python
_SENT = re.compile(r"(?<=[.!?])\s+")
```

`_SENT` finds the gap between sentences: whitespace that comes right after a `.`, `!` or `?`. `(?<=...)` is a *look-behind*: it checks the punctuation without including it in the match.

### Word lists

**Lines 8-16**

```python
STOP = set(
    """a an the is are was were be been am i me my we our you your it its of to in on for with at by from and or
    but if then so as that this these those do does did can could should would will shall may might have has had
    not no what which who whom how when where why about into than too very just also please want need get got like
    tell know there here any some more most other such only own same up down out over under again further once
    after before still ever never really actually hasn't haven't didn't don't isn't wasn't won't can't doesn't
    i'm it's what's there's let's
    put give take go come see look find send keep let try ask say make""".split()
)
```

`STOP` is the **stop-word** list: very common words (a, the, is, how, what...) that carry no topic meaning. They are skipped when matching. The text is written as one long string and `.split()` turns it into a list of words, and `set(...)` makes lookup fast. The last line adds very light verbs like 'put', 'give', 'take' that otherwise confused matching.

**Lines 18-19**

```python
# Words that appear in many questions but say nothing about the topic. They are used for ranking
# but are not treated as "must-have facts" by the evidence check.
```

A comment explaining the next list: words that appear in many questions but say nothing about the topic.

**Lines 20-25**

```python
GENERIC = set(
    """long much many window policy rule information info detail process procedure time thing stuff way work use make
    help question possible able mean day week month year hour minute two three four five six seven eight nine ten
    ago today yesterday tomorrow now soon old new first last next one something anything everything someone
    order parcel package item product purchase""".split()
)
```

`GENERIC` is the second list: words like 'long', 'much', 'week', 'order', 'item'. They are *kept* for ranking but are **not** treated as must-have facts by the evidence check. Without this list, a question like 'how many days...' would be rejected because the answer text never contains the word 'days' in that form.

### Splitting text into words

**Lines 28-29**

```python
def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())
```

`tokenize` turns text into a list of words. It lower-cases the text first, then `findall` returns every match of the word pattern.

### The stemmer

**Lines 32-33**

```python
def stem(word: str) -> str:
    """A deliberately tiny stemmer. It only needs to be consistent between documents and questions."""
```

`stem` cuts endings off a word so related forms look the same. The docstring says it is deliberately tiny: it only has to be **consistent** between documents and questions, not linguistically perfect.

**Lines 34-35**

```python
    if "'" in word:
        word = word.split("'")[0]
```

If the word has an apostrophe (like "don't"), keep only the part before it.

**Line 36**

```python
    if len(word) > 4:
```

Only words longer than 4 letters are shortened, so short words such as 'was' or 'bus' are left alone.

**Lines 37-38**

```python
        if word.endswith("ies"):
            word = word[:-3] + "y"
```

A word ending in 'ies' becomes '...y' (policies becomes policy).

**Lines 39-40**

```python
        elif word.endswith("ing") and len(word) > 5:
            word = _undouble(word[:-3])
```

A word ending in 'ing' (and long enough) loses it, then `_undouble` fixes doubled letters (shipping becomes ship).

**Lines 41-42**

```python
        elif word.endswith("ed"):
            word = _undouble(word[:-2])
```

A word ending in 'ed' loses it (refunded becomes refund).

**Lines 43-44**

```python
        elif word.endswith(("sses", "xes", "zes", "ches", "shes")):
            word = word[:-2]
```

Endings like 'sses', 'xes', 'zes', 'ches', 'shes' lose their last two letters (boxes becomes box).

**Lines 45-46**

```python
        elif word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
```

A plain plural 's' is dropped, but not for words ending 'ss' (so 'class' stays).

**Lines 47-48**

```python
    if len(word) > 8 and word.endswith("ment"):
        word = word[:-4]
```

Very long words ending in 'ment' lose it, so 'reimbursement' and 'reimbursed' meet at the same stem.

**Lines 49-50**

```python
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
```

A trailing 'e' is dropped, so 'damage' and 'damaged' both end up as 'damag'.

**Line 51**

```python
    return word
```

Returns the shortened word.

**Lines 54-57**

```python
def _undouble(w: str) -> str:
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "lsz":
        return w[:-1]
    return w
```

`_undouble`: if a word ends in a doubled letter (like 'pp' in 'shipp'), drop one of them, except for 'l', 's' and 'z' where doubles are normal ('cancell', 'class').

### Helpers built on top

**Lines 60-61**

```python
def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOP]
```

`content_tokens`: the words of a text with the stop words removed.

**Lines 64-65**

```python
def content_stems(text: str) -> list[str]:
    return [stem(t) for t in content_tokens(text)]
```

`content_stems`: the same, with each word stemmed. This is the form used for almost all matching.

**Lines 68-70**

```python
def is_wordlike(token: str) -> bool:
    """True for ordinary words; False for numbers, money and percentages."""
    return token[0].isalpha()
```

`is_wordlike`: true if the first character is a letter. Numbers, money and percentages (like '15%' or '$240') return false, so they are not treated as vocabulary.

**Lines 73-74**

```python
def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]
```

`sentences`: splits a paragraph into sentences using the sentence pattern, trims each, and drops empty ones.

**Lines 77-78**

```python
def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
```

`slug`: turns a title into a safe identifier, for example 'Damage Refund' becomes 'damage-refund'. Everything that is not a letter or digit becomes a hyphen, and extra hyphens are trimmed from the ends.

**Line 81**

```python
GENERIC_STEMS = {stem(w) for w in GENERIC}
```

`GENERIC_STEMS`: the generic word list, passed through the stemmer too, so it can be compared with stemmed question words. It sits at the bottom because `stem` must already exist when this line runs.


---
