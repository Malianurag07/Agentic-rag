# Part 7: The agent pipeline (answering one question)

This is the file that ties everything together. When a user asks a question, `Pipeline.ask()` runs it through **fixed stages**, and at certain points the agent **makes a choice**: trust the top result, ask the judge, ask the user, try the next candidate, or give up honestly.

```
question -> chit-chat? -> split into sub-questions -> for each sub-question:
    rewrite (typos, follow-ups)
    -> find candidate sections (hybrid retrieval)
    -> too weak a match? ......................... fallback search, else refuse
    -> only names the topic ('refund')? ........... ask the user which type
    -> confident? yes: take the top section / no: ask the judge to choose
    -> fetch its chunks -> evidence check
         passes ................................... write the answer, check it, cite the source
         fails .................................... soft-block it, try the next candidate (max 3)
    -> still nothing ............................. fallback search, else refuse
```

Safety is built in around this flow: a step limit, a time limit and loop detection (`Budget`), the permission filter (already in the store), and the removal of hostile sentences from retrieved text.

## `agentic_rag/pipeline.py`

The agent loop and its guardrails. It reads top to bottom roughly in the order things happen.

### File description and imports

**Lines 1-7**

```python
"""Online half of the pipeline (plan section 6): the agent loop with its guardrails (section 8).

Fixed stages, with the agent making choices inside them:

    prep -> permission filter -> score branches -> read score gap -> (LLM choice | ask user | fallback)
         -> fetch chunks -> evidence check -> (soft-block + retry | fallback) -> answer -> faithfulness check
"""
```

Describes the online half of the pipeline, 'fixed stages with the agent making choices inside them', and lists the stages in order.

**Lines 9-11**

```python
import re
import time
from dataclasses import dataclass, field
```

`re` (patterns), `time` (the clock, for the time limit) and the record tools.

**Lines 13-19**

```python
from .config import Config
from .embeddings import cosine
from .models import Candidate, Chunk
from .retrieval import Retriever, SmallToBigRetriever
from .store import Store
from .text import content_stems, content_tokens, tokenize
from .tools import sanitize, try_calculate
```

Imports every piece it coordinates: the settings, the similarity function, the data models, the two retrievers, the store, text helpers, and the two tools (injection filter and fee calculator).

### Greetings, exceptions and the Budget guardrail

**Line 21**

```python
GREETINGS = {"hi", "hello", "hey", "thanks", "thank", "you", "good", "morning", "evening", "bye"}
```

`GREETINGS`: the set of words that make a message small talk ('hi', 'thanks', 'bye'...).

**Lines 24-25**

```python
class BudgetExceeded(Exception):
    pass
```

`BudgetExceeded`: an error raised when the step limit or time limit is hit.

**Lines 28-29**

```python
class LoopDetected(Exception):
    pass
```

`LoopDetected`: an error raised when the agent tries to do exactly the same thing twice.

**Lines 32-33**

```python
class Budget:
    """Step limit, time limit and loop detection in one place."""
```

`Budget` is one object per question that enforces three safety rules at once.

**Lines 35-37**

```python
    def __init__(self, max_steps: int, seconds: float):
        self.max_steps, self.deadline = max_steps, time.time() + seconds
        self.steps, self.seen = 0, set()
```

Created with a maximum number of steps and a number of seconds. It records the **deadline** (now plus the seconds), a step counter, and a set of actions already done.

**Lines 39-41**

```python
    def spend(self, tool: str, key: str = "") -> None:
        if self.steps >= self.max_steps:
            raise BudgetExceeded(f"step limit of {self.max_steps} reached")
```

`spend(tool, key)` is called before every action. If the step count has reached the maximum, it raises an error.

**Lines 42-43**

```python
        if time.time() > self.deadline:
            raise BudgetExceeded("time budget used up")
```

If the deadline has passed, it raises an error.

**Lines 44-45**

```python
        if (tool, key) in self.seen:
            raise LoopDetected(f"{tool}({key!r}) already ran")
```

If this exact action (same tool, same input) was already done, it raises `LoopDetected`: the agent is going in circles.

**Lines 46-47**

```python
        self.seen.add((tool, key))
        self.steps += 1
```

Otherwise the action is recorded and the step count goes up by one.

### The Result record

**Lines 50-54**

```python
@dataclass
class Result:
    question: str
    status: str                       # answered | refused | clarify
    answer: str
```

`Result` is what `ask()` returns. `status` is one of `answered`, `refused` or `clarify` (the system asks the user something); `answer` is the text shown to the user.

**Lines 55-57**

```python
    branch_id: str | None = None
    citations: list = field(default_factory=list)
    candidate_ids: list = field(default_factory=list)
```

`branch_id` is the section the answer came from; `citations` lists sources; `candidate_ids` lists the top candidate sections found.

**Lines 58-62**

```python
    tries: int = 0
    steps: int = 0
    llm_calls: int = 0
    used_fallback: bool = False
    elapsed: float = 0.0
```

`tries` (sections tried), `steps` (actions used), `llm_calls` (judgment calls), `used_fallback` (whether the fallback search was needed) and `elapsed` (seconds).

**Lines 63-64**

```python
    trace: list = field(default_factory=list)
    branch_ids: list = field(default_factory=list)    # one per sub-question for multi-part questions
```

`trace` is a step-by-step log of what happened, for debugging; `branch_ids` holds one section per sub-question for multi-part questions.

### Creating the pipeline

**Lines 67-69**

```python
class Pipeline:
    def __init__(self, store: Store, llm, config: Config | None = None):
        self.store, self.llm, self.config = store, llm, config or Config()
```

A `Pipeline` is built from a store (the data), an `llm` (the judge: rule-based or Claude) and settings.

**Line 70**

```python
        self._retrievers: dict = {}
```

A cache of retrievers, so the search structures are built once per user group.

**Lines 71-72**

```python
        kind = getattr(store.embedder, "kind", "")
        self.hybrid = self.config.retrieval == "hybrid" or (self.config.retrieval == "auto" and kind == "onnx-minilm-l6-v2")
```

Decides whether to use **hybrid retrieval**: either the settings force it, or they say 'auto' and the stored embeddings come from the neural model. Otherwise the older word-overlap ranking is used.

### ask(): the public entry point

**Lines 74-80**

```python
    # ---- public entry point -------------------------------------------------------------------------------------------
    def ask(self, question: str, groups=("public",), session: dict | None = None) -> Result:
        t0 = time.time()
        session = session if session is not None else {}
        calls_before = self.llm.calls
        trace: list = []
        steps = 0
```

`ask` answers one question for a user (their groups) within a session (a notebook for remembering the last topic). It starts a clock, makes sure there is a session, notes how many judgment calls had happened before, and prepares an empty trace and a step counter.

**Lines 82-85**

```python
        if self._is_chitchat(question):
            topics = ", ".join(t.name.lower() for t in self.store.topics(groups))
            return Result(question, "answered", f"Hello! I can help with: {topics}. What would you like to know?",
                          elapsed=time.time() - t0, trace=[{"event": "chitchat shortcut"}])
```

If the message is small talk, it answers immediately with a greeting listing the visible topics, and no search is done.

**Lines 87-89**

```python
        subs = self._plan(question)
        if len(subs) > 1:
            trace.append({"event": "planner", "sub_questions": subs})
```

Splits the question into sub-questions (a normal question is just one). If there are several, it logs the plan.

**Lines 90-94**

```python
        parts = []
        for sub in subs:
            budget = Budget(self.config.max_agent_steps, self.config.time_budget_seconds)
            parts.append(self._run_single(sub, groups, session, trace, budget))
            steps += budget.steps
```

For each sub-question: a **fresh budget** is created, the sub-question is run through all the stages, and the steps used are added up.

**Lines 96-97**

```python
        if len(parts) == 1:
            res = parts[0]
```

With one sub-question, its result is the answer.

**Lines 98-100**

```python
        else:
            answered = [p for p in parts if p.status == "answered"]
            body = "\n\n".join(f"{s}: {p.answer}" for s, p in zip(subs, parts))
```

With several: collects the ones that were answered, and builds one answer where each sub-question is followed by its answer.

**Lines 101-105**

```python
            res = Result(question, "answered" if answered else "refused", body,
                         branch_id=parts[0].branch_id, branch_ids=[p.branch_id for p in parts],
                         citations=[c for p in parts for c in p.citations],
                         candidate_ids=parts[0].candidate_ids, tries=sum(p.tries for p in parts),
                         used_fallback=any(p.used_fallback for p in parts))
```

Builds the combined Result: 'answered' if at least one part was answered, otherwise 'refused'; with the section of each part, all citations, the first part's candidates, the total tries, and whether any part used the fallback.

**Lines 106-107**

```python
        res.question, res.steps, res.trace = question, steps, trace
        res.llm_calls, res.elapsed = self.llm.calls - calls_before, time.time() - t0
```

Records the original question, total steps and trace, and the number of judgment calls and time taken.

**Lines 108-110**

```python
        if not res.branch_ids:
            res.branch_ids = [res.branch_id]
        return res
```

Makes sure `branch_ids` is a list, then returns the result.

### Small talk and splitting multi-part questions

**Lines 112-116**

```python
    # ---- planning (plan 7.4) ---------------------------------------------------------------------------------------------
    @staticmethod
    def _is_chitchat(q: str) -> bool:
        toks = tokenize(q)
        return bool(toks) and all(t in GREETINGS for t in toks)
```

`_is_chitchat` is true when the message has words and **all** of them are greetings.

**Line 118**

```python
    SPLIT = re.compile(r",?\s+and\s+(?=(?:what|how|who|which|when|where|why|can|could|is|are|do|does|must|should)\b)|;\s+|\?\s+(?=(?:what|how|who|which|when|where|why|can|could|is|are|do|does|must|should)\b)", re.I)
```

`SPLIT` is the pattern that finds where to cut a multi-part question: ', and' followed by a question word (what, how, who, which, when, where, why, can, is...); a semicolon; or a question mark followed by a new question word. Plain 'and' in the middle of a phrase is not a cut.

**Lines 120-124**

```python
    @classmethod
    def _plan(cls, question: str) -> list[str]:
        """Split a multi-part question into sub-questions, each answered on its own (plan 7.4).
        "Compare X and Y" becomes two questions; so does "A?, and what B?". A short fragment that points back with
        "it", "that" or "this" keeps the first part as context."""
```

`_plan` returns the list of sub-questions. The docstring: 'compare X and Y' becomes two; a short fragment that points back ('it', 'that') keeps the first part for context.

**Lines 125-127**

```python
        m = re.match(r"\s*compare\s+(.+?)\s+(?:and|with|to|vs\.?|versus)\s+(.+?)\s*[?.!]*\s*$", question, re.I)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
```

Handles 'Compare A and B' (also 'with', 'to', 'vs', 'versus') by returning A and B as two questions.

**Lines 128-130**

```python
        parts = [p.strip(" ?.,;") for p in cls.SPLIT.split(question) if p and p.strip(" ?.,;")]
        if len(parts) < 2:
            return [question]
```

Otherwise splits with the pattern, trims punctuation, drops empty pieces. If fewer than two pieces, the question is not multi-part and is returned whole.

**Lines 131-135**

```python
        subs = [parts[0]]
        for part in parts[1:]:
            pointer = re.search(r"\b(it|that|this|they|them|those)\b", part, re.I)
            subs.append(f"{part} ({parts[0]})" if pointer or len(content_stems(part)) < 3 else part)
        return subs
```

The first piece stays as it is. Each later piece that points back ('it', 'that', 'they'...) or is too short (under 3 content words) gets the first piece added in brackets as context; other pieces stay as they are.

### _run_single(): one question through the stages

**Lines 137-141**

```python
    # ---- one question through the stages -----------------------------------------------------------------------------------
    def _run_single(self, question: str, groups, session: dict, trace: list, budget: Budget) -> Result:
        cfg, store = self.config, self.store
        raw_words, df, n_docs = store.vocab(groups)
        log = lambda **kw: trace.append({"q": question, **kw})
```

The heart of the agent. It reads the settings and store, gets the vocabulary of the visible content (`df` counts how many chunks contain each word), and defines `log`, a shortcut that adds an entry to the trace.

**Lines 142-144**

```python
        try:
            q = self.llm.rewrite(question, session.get("last_topic"), raw_words)
            log(event="query prep", prepared=q)
```

Everything runs inside a `try` so that the safety errors can be caught at the end. **Stage 1, rewrite**: repair typos and fill in follow-ups, and log the result.

**Lines 146-149**

```python
            budget.spend("score_branches", q)
            cands = self._rank(q, groups)
            log(event="score branches", top=[(c.branch.id, round(c.score, 3)) for c in cands[:5]])
            ids = [c.branch.id for c in cands[:3]]
```

**Stage 2, find candidates**: spend a step, rank the sections (hybrid retrieval or older ranking), log the top five, and remember the ids of the top three.

**Lines 150-151**

```python
            floor_value = (cands[0].score if self.hybrid else cands[0].score) if cands else 0.0
            floor = cfg.dense_floor if self.hybrid else cfg.score_floor
```

Decides how to measure 'is the best match good enough': in hybrid mode it uses the meaning-similarity floor (0.30), otherwise the older floor (0.12). (Line 150 has a leftover redundancy: both branches of its `if` give the same value, `cands[0].score`; that is harmless but could be simplified.)

**Lines 152-154**

```python
            if not cands or floor_value < floor:
                log(event="all scores low", best=round(floor_value, 3), floor=floor)
                return self._fallback(q, question, groups, session, trace, budget, ids, tries=0)
```

If there are no candidates or the best score is below the floor, nothing in the knowledge base matches: log it and go to the **fallback**.

**Lines 156-161**

```python
            if self._topic_only(q, cands[0]):
                topic = store.branches[cands[0].branch.parent_id]
                names = [b.name for b in store.leaves(groups) if b.parent_id == topic.id]
                log(event="topic-only question, asking user", topic=topic.id)
                return Result(question, "clarify", f"Which {topic.name.lower()} question do you mean: " + ", ".join(names) + "?",
                              candidate_ids=ids)
```

**Stage 3, topic-only question**: if the question names only a topic ('refund', 'billing') without pointing at one type, there is no way to choose, so it returns a clarifying question that lists the available types.

**Lines 163-164**

```python
            gap = cands[0].score - (cands[1].score if len(cands) > 1 else 0.0)
            confident = cands[0].agree if self.hybrid else gap >= cfg.score_gap_clear
```

**Stage 4, confidence**: `gap` is the lead of the best candidate over the second. In hybrid mode, `confident` is true if the two rankers agree on the top section (or it leads clearly); in older mode, if the gap is large enough.

**Lines 165-167**

```python
            if confident:
                pick = cands[0]
                log(event="clear gap, LLM choice skipped", gap=round(gap, 3), branch=pick.branch.id)
```

If confident, the top candidate is taken and the judge is **not** asked (saves a call).

**Lines 168-171**

```python
            else:
                budget.spend("choose", q)
                choice = self.llm.choose(q, cands[:cfg.llm_candidates_shown])
                log(event="LLM choice", gap=round(gap, 3), branch=choice.branch_id, reason=choice.reason)
```

Otherwise a step is spent and the judge is asked to choose among the top five candidates; the choice is logged.

**Lines 172-177**

```python
                if choice.branch_id is None:
                    unknown = self._unknown_terms(q, df)
                    if unknown:
                        log(event="no clear branch and the question uses words the knowledge base has never seen: refusing, not asking",
                            unknown=unknown)
                        return self._fallback(q, question, groups, session, trace, budget, ids, tries=0)
```

If the judge could not choose, and the question contains words the knowledge base has never seen (like 'car' in a policy manual), the question is probably outside the documents: refuse via the fallback instead of asking the user.

**Lines 178-179**

```python
                    return Result(question, "clarify", choice.ask or "Could you tell me a bit more?",
                                  candidate_ids=ids, llm_calls=0)
```

Otherwise the system returns a clarifying question to the user (the judge's, or a default).

**Line 180**

```python
                pick = next((c for c in cands if c.branch.id == choice.branch_id), cands[0])
```

If the judge chose a section, that candidate is used (the top one if the id is somehow not found).

**Lines 182-184**

```python
            tries = 0
            blocked: dict[str, str] = {}
            while pick is not None and tries < cfg.max_branch_tries:
```

**Stage 5, try the chosen section**. `tries` counts attempts; `blocked` remembers rejected sections. It loops while there is a candidate and fewer than 3 tries have been used.

**Lines 185-187**

```python
                tries += 1
                budget.spend("get_chunks", pick.branch.id)
                chunks = self._safe_chunks(pick.branch.id, groups, trace, question)
```

Counts the try, spends a step, and fetches the section's chunks, cleaned of any hostile sentences.

**Lines 188-189**

```python
                siblings = [(c.branch, self._safe_chunks(c.branch.id, groups, [], question))
                            for c in cands[:3] if c.branch.id != pick.branch.id and c.branch.id not in blocked]
```

Also fetches the chunks of up to two neighbouring top candidates (not the chosen one, not already rejected), so the judge can compare them.

**Lines 190-193**

```python
                budget.spend("evidence_check", pick.branch.id)
                check = self.llm.check(q, pick.branch, chunks, siblings, df, n_docs, store.branches.get(pick.branch.parent_id))
                log(event="evidence check", branch=pick.branch.id, ok=check.ok, reason=check.reason,
                    ratio=round(check.ratio, 2), quotes=check.supports[:3])
```

**Evidence check**: spends a step, asks the judge whether the text really supports the question, and logs the verdict with its ratio and quotes.

**Lines 194-196**

```python
                relaxed = (not check.ok and confident and not check.unknown and check.ratio >= cfg.evidence_ratio_confident)
                if relaxed:
                    log(event="evidence check relaxed: retrieval had a clear lead", branch=pick.branch.id, ratio=round(check.ratio, 2))
```

A **relaxed pass**: if the check failed, but retrieval was confident, there are no unknown question words, and at least 40% of the facts are supported, the answer is accepted anyway. This lowered bar is what makes weakly supported answers possible, and it is logged.

**Lines 197-198**

```python
                if check.ok or relaxed:
                    return self._respond(question, q, pick.branch, chunks, session, trace, budget, ids, tries, False)
```

If the check passed (or was relaxed), the answer is written by `_respond`.

**Lines 199-201**

```python
                blocked[pick.branch.id] = check.reason          # soft block: this question only, kept in the list
                log(event="soft-block", branch=pick.branch.id, reason=check.reason)
                pick = next((c for c in cands if c.branch.id not in blocked and c.score >= cfg.score_floor), None)
```

Otherwise the section is **soft-blocked**: remembered as rejected for this question only (it stays in the list), and the next candidate not blocked and above the similarity floor is picked. If there is none, the loop ends.

**Lines 203-204**

```python
            log(event="tries exhausted or no branch left", blocked=list(blocked))
            return self._fallback(q, question, groups, session, trace, budget, ids, tries)
```

If the loop ended without an answer, log it and go to the fallback.

**Lines 205-207**

```python
        except (BudgetExceeded, LoopDetected) as exc:
            log(event="guardrail", detail=str(exc))
            return Result(question, "refused", f"I couldn't finish within my limits ({exc}). Please rephrase or narrow the question.")
```

If the budget or loop guard fired anywhere above, log it and return a polite refusal that mentions the limit.

### Helpers for ranking

**Lines 209-214**

```python
    # ---- ranking (plan 6.3): path-aware leaf scoring --------------------------------------------------------------------------
    @staticmethod
    def _unknown_terms(q: str, df) -> list[str]:
        from .text import GENERIC_STEMS, is_wordlike, stem
        return [stem(t) for t in content_tokens(q)
                if is_wordlike(t) and stem(t) not in GENERIC_STEMS and df.get(stem(t), 0) == 0]
```

`_unknown_terms` lists question words that appear nowhere in the visible knowledge base (not numbers, not generic words). Used to tell 'vague question' from 'question outside the documents'. (The comment above it, about ranking, belongs to the ranking functions below; it was left in a slightly misleading place.)

**Lines 216-220**

```python
    def _stems_of(self, chunk: Chunk) -> set:
        cache = self.__dict__.setdefault("_stem_cache", {})
        if chunk.id not in cache:
            cache[chunk.id] = set(content_stems(chunk.embed_text()))
        return cache[chunk.id]
```

`_stems_of` returns the word stems of a chunk, remembering them in a cache so each chunk is only processed once.

**Lines 222-225**

```python
    def _retriever(self, groups):
        key = tuple(sorted(groups))
        if key not in self._retrievers:
            chunks = self.store.visible_chunks(groups)
```

`_retriever` builds, once per user group, the search object over the chunks the user may see.

**Lines 226-229**

```python
            if self.config.parent_child and hasattr(self.store.embedder, "embed_many"):
                self._retrievers[key] = SmallToBigRetriever(chunks, self.store.embedder, mode="hybrid")
            else:
                self._retrievers[key] = Retriever(chunks, self.store.embedder, mode="hybrid")
```

If parent-child matching is on and the embedder supports batches (the neural one), it builds a `SmallToBigRetriever` (match on small units, return whole chunks); otherwise a plain hybrid `Retriever`.

**Line 230**

```python
        return self._retrievers[key]
```

Returns the cached retriever.

### Hybrid ranking of sections

**Lines 232-235**

```python
    def _rank_hybrid(self, q: str, groups) -> list[Candidate]:
        """Hybrid retrieval (plan 6.3): chunks ranked by meaning (neural embedding) and by words (BM25), merged by reciprocal rank
        fusion; a section is ranked by its best chunk. Each candidate carries the best meaning-similarity of its chunks as its
        score, and `agree` says whether both rankers put the top section first."""
```

`_rank_hybrid` ranks sections using the hybrid retriever. Each candidate's `score` is the best meaning similarity among its chunks, and `agree` records whether both rankers put the top section first.

**Lines 236-238**

```python
        retr = self._retriever(groups)
        detailed = retr.rank_chunks_detailed(q, k=40)
        leaves = {l.id: l for l in self.store.leaves(groups)}
```

Gets the top 40 chunks with detailed scores, and the visible sections (leaves), keyed by id.

**Lines 239-243**

```python
        order, dense, d_rank, s_rank = [], {}, {}, {}
        for chunk, _fused, cos, rd, rs in detailed:
            b = chunk.branch_id
            if b not in leaves:
                continue
```

Prepares ordered section list and three dictionaries (best meaning similarity, best meaning rank, best word rank per section). For each chunk, skip those whose section is not visible.

**Lines 244-248**

```python
            if b not in order:
                order.append(b)
            dense[b] = max(dense.get(b, 0.0), cos)
            d_rank[b] = min(d_rank.get(b, 10 ** 6), rd)
            s_rank[b] = min(s_rank.get(b, 10 ** 6), rs)
```

A section is added to the order the first time its chunk appears (so sections are ordered by their best chunk). For each section it keeps the highest meaning similarity and the best (lowest) ranks.

**Lines 249-250**

```python
        if not order:
            return []
```

If there are no sections, return an empty list.

**Lines 251-255**

```python
        top = order[0]
        best_dense = min(d_rank, key=d_rank.get)
        best_word = min(s_rank, key=s_rank.get)
        others = [dense[b] for b in order[1:]]
        lead = dense[top] - (max(others) if others else 0.0)
```

`top` is the first section; `best_dense` and `best_word` are the sections with the best meaning rank and best word rank; `lead` is how far the top section's meaning similarity is ahead of the next-best.

**Lines 256-257**

```python
        out = [Candidate(leaves[b], dense[b], 0.0, dense[b]) for b in order[:self.config.top_k_branches]]
        out[0].agree = (top == best_dense and top == best_word) or lead >= self.config.dense_gap_clear
```

Creates the Candidate list (up to the configured number), and marks the top candidate's `agree` as true if both rankers agree on it or it leads clearly.

**Line 258**

```python
        return out
```

Returns the candidates.

**Lines 260-263**

```python
    def _rank(self, q: str, groups) -> list[Candidate]:
        if self.hybrid:
            return self._rank_hybrid(q, groups)
        return self._rank_legacy(q, groups)
```

`_rank` chooses between the hybrid ranking and the older one.

### The older ranking (for small hash-embedded knowledge bases)

**Lines 265-270**

```python
    def _rank_legacy(self, q: str, groups) -> list[Candidate]:
        """Score each subsection leaf from two angles and keep the better one: how well its title and opening words match,
        and how well its single best chunk matches (embedding plus shared words). Facts that live deep in a section's body
        are therefore found even when its title says nothing about them. A small weight from the parent topic is added."""
        qvec = self.store.embedder.embed(q)
        qs = set(content_stems(q))
```

`_rank_legacy`: used when embeddings are not neural. Each section is scored two ways and the better one kept: how well its title matches, and how well its best chunk matches (meaning plus shared words). It embeds the question and takes the question's word stems.

**Line 271**

```python
        topic_scores = {t.id: cosine(qvec, t.embedding) for t in self.store.topics(groups)}
```

Scores every topic against the question.

**Lines 272-277**

```python
        best_chunk: dict[str, float] = {}
        for c in self.store.visible_chunks(groups):
            overlap = len(qs & self._stems_of(c)) / (len(qs) or 1)
            h = cosine(qvec, c.embedding) + self.config.keyword_weight * overlap
            if h > best_chunk.get(c.branch_id, 0.0):
                best_chunk[c.branch_id] = h
```

For every visible chunk: a score of its similarity plus a bonus for the share of question words it contains; keeps the best chunk score for each section.

**Lines 278-285**

```python
        w = self.config.topic_weight
        out = []
        for leaf in self.store.leaves(groups):
            ts = topic_scores.get(leaf.parent_id, 0.0)
            cs = max(cosine(qvec, leaf.embedding), best_chunk.get(leaf.id, 0.0))
            out.append(Candidate(leaf, w * ts + (1 - w) * cs, ts, cs))
        out.sort(key=lambda c: c.score, reverse=True)
        return out[:self.config.top_k_branches]
```

For every section: a score that blends its parent topic's score with the better of its own title score and its best chunk. Sorts best first and keeps the top candidates.

### Spotting questions that only name a topic

**Lines 287-292**

```python
    def _topic_only(self, q: str, top: Candidate) -> bool:
        """True when every word in the question is already covered by the topic itself, so no word
        points at one particular type (e.g. just "refund" or "billing")."""
        topic = self.store.branches.get(top.branch.parent_id)
        if topic is None:
            return False
```

`_topic_only` is true when the question's words are all covered by the topic itself, so nothing points to one particular type. It finds the parent topic; with no parent there is nothing to ask.

**Lines 293-295**

```python
        words = set(content_stems(q))
        if not words:
            return False
```

Takes the question's word stems; an empty question is not topic-only.

**Lines 296-298**

```python
        siblings = [b for b in self.store.branches.values() if b.parent_id == topic.id and b.status == "active"]
        if len(siblings) <= 1:
            return False                                    # nothing to choose between
```

Finds the active siblings under that topic. With one or none there is nothing to choose between.

**Lines 299-303**

```python
        topic_words = set(content_stems(topic.search_text()))
        topic_name = set(content_stems(topic.name))
        leaf = top.branch
        leaf_specific = (set(content_stems(leaf.name)) - topic_name) | (set(content_stems(leaf.search_text())) - topic_words)
        return words <= topic_words and not (words & leaf_specific)
```

Computes the topic's words and the words that are **specific** to the candidate type (its name minus the topic name, plus its other text minus the topic's words). The question is topic-only if all its words are topic words **and** none of them is specific to the type.

### Fetching chunks safely

**Lines 305-308**

```python
    def _safe_chunks(self, branch_id: str, groups, trace: list, question: str) -> list[Chunk]:
        """Fetch chunks and strip anything that tries to give the AI orders (plan section 8)."""
        out = []
        for c in self.store.chunks_for(branch_id, groups)[:self.config.chunks_to_llm]:
```

`_safe_chunks` fetches the chunks of a section (up to 6), and cleans each one.

**Lines 309-312**

```python
            clean, removed = sanitize(c.text)
            if removed:
                trace.append({"q": question, "event": "prompt-injection removed", "chunk": c.id, "removed": removed})
            out.append(Chunk(**{**c.__dict__, "text": clean}))
```

Passes the text through the injection filter. If any sentences were removed, the removal is logged. A copy of the chunk with the cleaned text is used, so the stored original is never changed.

**Line 313**

```python
        return out
```

Returns the cleaned chunks.

### The fallback: when the main route fails

**Lines 315-320**

```python
    # ---- fallback path (plan 6.5) --------------------------------------------------------------------------------------------
    def _fallback(self, q, question, groups, session, trace, budget, ids, tries) -> Result:
        cfg = self.config
        raw_words, df, n_docs = self.store.vocab(groups)
        try:
            budget.spend("search_all_chunks", q)
```

`_fallback` is the safety net for when ranking or checking failed. It reads the vocabulary and spends a step on a search over **all** chunks.

**Lines 321-324**

```python
            if self.hybrid:
                detailed = self._retriever(groups).rank_chunks_detailed(q, k=cfg.fallback_top_k)
                trace.append({"q": question, "event": "fallback hybrid search",
                              "top": [(c.id, round(cos, 3)) for c, _f, cos, _a, _b in detailed[:3]]})
```

In hybrid mode: runs the hybrid search over every chunk the user may see and logs the top three.

**Lines 325-327**

```python
                if detailed and detailed[0][2] >= cfg.dense_floor:
                    branch = self.store.branches[detailed[0][0].branch_id]
                    chunks = self._safe_chunks(branch.id, groups, trace, question)
```

If the best chunk's meaning similarity reaches the floor, it takes that chunk's section and fetches its cleaned chunks.

**Lines 328-331**

```python
                    budget.spend("evidence_check", "fallback:" + branch.id)
                    check = self.llm.check(q, branch, chunks, [], df, n_docs, self.store.branches.get(branch.parent_id))
                    trace.append({"q": question, "event": "fallback evidence check", "branch": branch.id,
                                  "ok": check.ok, "reason": check.reason})
```

Spends a step, runs the evidence check on it, and logs the verdict.

**Lines 332-333**

```python
                    if check.ok or (not check.unknown and check.ratio >= cfg.evidence_ratio_confident):
                        return self._respond(question, q, branch, chunks, session, trace, budget, ids, tries, True)
```

If the check passes, or fails only mildly (no unknown words and at least 40% supported), it answers.

**Line 334**

```python
                raise LoopDetected("no reliable match")       # handled below as a plain refusal
```

Otherwise it raises `LoopDetected` as a handy way to jump to the plain refusal at the bottom.

**Lines 335-336**

```python
            qvec = self.store.embedder.embed(q)
            qs = set(content_stems(q))
```

Older mode: embeds the question and takes its word stems.

**Lines 337-342**

```python
            scored = []
            for c, cos in self.store.search_chunks(qvec, groups, 50):
                overlap = len(qs & set(content_stems(c.text))) / (len(qs) or 1)
                scored.append((cos + cfg.keyword_weight * overlap, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = [(s, c) for s, c in scored[:cfg.fallback_top_k] if s >= cfg.fallback_floor]
```

Scores the 50 most similar chunks by similarity plus a bonus for shared words, and keeps the top few above the fallback floor.

**Lines 343-344**

```python
            trace.append({"q": question, "event": "fallback direct search",
                          "top": [(c.id, round(s, 3)) for s, c in top[:3]]})
```

Logs the best of them.

**Lines 345-353**

```python
            if top:
                branch = self.store.branches[top[0][1].branch_id]
                chunks = self._safe_chunks(branch.id, groups, trace, question)
                budget.spend("evidence_check", "fallback:" + branch.id)
                check = self.llm.check(q, branch, chunks, [], df, n_docs, self.store.branches.get(branch.parent_id))
                trace.append({"q": question, "event": "fallback evidence check", "branch": branch.id,
                              "ok": check.ok, "reason": check.reason})
                if check.ok:
                    return self._respond(question, q, branch, chunks, session, trace, budget, ids, tries, True)
```

If any chunk qualified, the same flow: take its section, check the evidence strictly, and answer only if the check passes.

**Lines 354-355**

```python
        except (BudgetExceeded, LoopDetected) as exc:
            trace.append({"q": question, "event": "guardrail", "detail": str(exc)})
```

If a safety limit or the jump above was triggered, it is logged and the code falls through to refusing.

**Lines 356-360**

```python
        names = ", ".join(self.store.branches[i].name for i in ids) or "none"
        return Result(question, "refused",
                      "I couldn't find a reliable answer in the knowledge base. "
                      f"Closest topics I checked: {names}. Could you rephrase or add more detail?",
                      candidate_ids=ids, tries=tries, used_fallback=True)
```

Builds the honest refusal: 'I couldn't find a reliable answer in the knowledge base', naming the closest topics it checked and inviting a rephrase.

### Writing the answer

**Lines 362-366**

```python
    # ---- answer + faithfulness (plan 6.8) -----------------------------------------------------------------------------------------
    def _respond(self, question, q, branch, chunks, session, trace, budget, ids, tries, used_fallback) -> Result:
        budget.spend("answer", branch.id)
        body = self.llm.answer(q, branch, chunks)
        bad = self.llm.unsupported(body, chunks)
```

`_respond` produces the final answer. It spends a step, asks the judge to write the answer from the retrieved chunks, then asks which sentences are unsupported.

**Lines 367-369**

```python
        if bad:
            trace.append({"q": question, "event": "faithfulness: removed", "sentences": bad})
            body = " ".join(s for s in re.split(r"(?<=[.!?])\s+", body) if s not in bad)
```

If any sentences are unsupported, they are logged and removed from the answer.

**Lines 370-372**

```python
        if not body.strip():
            return Result(question, "refused", "I found related text but could not support an answer from it.",
                          branch_id=branch.id, candidate_ids=ids, tries=tries, used_fallback=used_fallback)
```

If nothing is left after the removal, the answer is a refusal: related text was found but could not support an answer.

**Lines 373-377**

```python
        calc = try_calculate(question)
        if calc:
            budget.spend("calculator", calc[0])
            body += f"\n\nCalculation: {calc[0]} = {calc[1]:.2f}"
            trace.append({"q": question, "event": "calculator", "expression": calc[0], "result": calc[1]})
```

If the question is a fee calculation, the safe calculator is run and the result is appended as a 'Calculation:' line; the step and result are logged.

**Lines 378-379**

```python
        sources = sorted({c.source for c in chunks if c.source})
        cite = f"{branch.path}" + (f" ({', '.join(sources)})" if sources else "")
```

Builds the **citation**: the section's breadcrumb path plus the document names it came from.

**Line 380**

```python
        session["last_topic"] = self.store.branches[branch.parent_id].name if branch.parent_id else branch.name
```

Remembers the section's parent topic in the session, so a follow-up like 'what about the late one?' knows what 'it' refers to.

**Lines 381-382**

```python
        return Result(question, "answered", f"{body}\n\nSource: {cite}", branch_id=branch.id, citations=[cite],
                      candidate_ids=ids, tries=tries, used_fallback=used_fallback)
```

Returns the final `answered` Result: the answer text, a 'Source:' line, the section, the citation, the candidate sections, the number of tries and whether the fallback was used.


---
