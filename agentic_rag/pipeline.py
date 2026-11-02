"""Online half of the pipeline (plan section 6): the agent loop with its guardrails (section 8).

Fixed stages, with the agent making choices inside them:

    prep -> permission filter -> score branches -> read score gap -> (LLM choice | ask user | fallback)
         -> fetch chunks -> evidence check -> (soft-block + retry | fallback) -> answer -> faithfulness check
"""

import re
import time
from dataclasses import dataclass, field

from .config import Config
from .embeddings import cosine
from .models import Candidate, Chunk
from .store import Store
from .text import content_stems, content_tokens, tokenize
from .tools import sanitize, try_calculate

GREETINGS = {"hi", "hello", "hey", "thanks", "thank", "you", "good", "morning", "evening", "bye"}


class BudgetExceeded(Exception):
    pass


class LoopDetected(Exception):
    pass


class Budget:
    """Step limit, time limit and loop detection in one place."""

    def __init__(self, max_steps: int, seconds: float):
        self.max_steps, self.deadline = max_steps, time.time() + seconds
        self.steps, self.seen = 0, set()

    def spend(self, tool: str, key: str = "") -> None:
        if self.steps >= self.max_steps:
            raise BudgetExceeded(f"step limit of {self.max_steps} reached")
        if time.time() > self.deadline:
            raise BudgetExceeded("time budget used up")
        if (tool, key) in self.seen:
            raise LoopDetected(f"{tool}({key!r}) already ran")
        self.seen.add((tool, key))
        self.steps += 1


@dataclass
class Result:
    question: str
    status: str                       # answered | refused | clarify
    answer: str
    branch_id: str | None = None
    citations: list = field(default_factory=list)
    candidate_ids: list = field(default_factory=list)
    tries: int = 0
    steps: int = 0
    llm_calls: int = 0
    used_fallback: bool = False
    elapsed: float = 0.0
    trace: list = field(default_factory=list)
    branch_ids: list = field(default_factory=list)    # one per sub-question for multi-part questions


class Pipeline:
    def __init__(self, store: Store, llm, config: Config | None = None):
        self.store, self.llm, self.config = store, llm, config or Config()

    # ---- public entry point -------------------------------------------------------------------------------------------
    def ask(self, question: str, groups=("public",), session: dict | None = None) -> Result:
        t0 = time.time()
        session = session if session is not None else {}
        calls_before = self.llm.calls
        trace: list = []
        steps = 0

        if self._is_chitchat(question):
            topics = ", ".join(t.name.lower() for t in self.store.topics(groups))
            return Result(question, "answered", f"Hello! I can help with: {topics}. What would you like to know?",
                          elapsed=time.time() - t0, trace=[{"event": "chitchat shortcut"}])

        subs = self._plan(question)
        if len(subs) > 1:
            trace.append({"event": "planner", "sub_questions": subs})
        parts = []
        for sub in subs:
            budget = Budget(self.config.max_agent_steps, self.config.time_budget_seconds)
            parts.append(self._run_single(sub, groups, session, trace, budget))
            steps += budget.steps

        if len(parts) == 1:
            res = parts[0]
        else:
            answered = [p for p in parts if p.status == "answered"]
            body = "\n\n".join(f"{s}: {p.answer}" for s, p in zip(subs, parts))
            res = Result(question, "answered" if answered else "refused", body,
                         branch_id=parts[0].branch_id, branch_ids=[p.branch_id for p in parts],
                         citations=[c for p in parts for c in p.citations],
                         candidate_ids=parts[0].candidate_ids, tries=sum(p.tries for p in parts),
                         used_fallback=any(p.used_fallback for p in parts))
        res.question, res.steps, res.trace = question, steps, trace
        res.llm_calls, res.elapsed = self.llm.calls - calls_before, time.time() - t0
        if not res.branch_ids:
            res.branch_ids = [res.branch_id]
        return res

    # ---- planning (plan 7.4) ---------------------------------------------------------------------------------------------
    @staticmethod
    def _is_chitchat(q: str) -> bool:
        toks = tokenize(q)
        return bool(toks) and all(t in GREETINGS for t in toks)

    @staticmethod
    def _plan(question: str) -> list[str]:
        m = re.match(r"\s*compare\s+(.+?)\s+(?:and|with|to|vs\.?|versus)\s+(.+?)\s*[?.!]*\s*$", question, re.I)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
        return [question]

    # ---- one question through the stages -----------------------------------------------------------------------------------
    def _run_single(self, question: str, groups, session: dict, trace: list, budget: Budget) -> Result:
        cfg, store = self.config, self.store
        raw_words, df, n_docs = store.vocab(groups)
        log = lambda **kw: trace.append({"q": question, **kw})
        try:
            q = self.llm.rewrite(question, session.get("last_topic"), raw_words)
            log(event="query prep", prepared=q)

            budget.spend("score_branches", q)
            cands = self._rank(q, groups)
            log(event="score branches", top=[(c.branch.id, round(c.score, 3)) for c in cands[:5]])
            ids = [c.branch.id for c in cands[:3]]
            if not cands or cands[0].score < cfg.score_floor:
                log(event="all scores low", best=cands[0].score if cands else 0)
                return self._fallback(q, question, groups, session, trace, budget, ids, tries=0)

            if self._topic_only(q, cands[0]):
                topic = store.branches[cands[0].branch.parent_id]
                names = [b.name for b in store.leaves(groups) if b.parent_id == topic.id]
                log(event="topic-only question, asking user", topic=topic.id)
                return Result(question, "clarify", f"Which {topic.name.lower()} question do you mean: " + ", ".join(names) + "?",
                              candidate_ids=ids)

            gap = cands[0].score - (cands[1].score if len(cands) > 1 else 0.0)
            if gap >= cfg.score_gap_clear:
                pick = cands[0]
                log(event="clear gap, LLM choice skipped", gap=round(gap, 3), branch=pick.branch.id)
            else:
                budget.spend("choose", q)
                choice = self.llm.choose(q, cands[:cfg.llm_candidates_shown])
                log(event="LLM choice", gap=round(gap, 3), branch=choice.branch_id, reason=choice.reason)
                if choice.branch_id is None:
                    return Result(question, "clarify", choice.ask or "Could you tell me a bit more?",
                                  candidate_ids=ids, llm_calls=0)
                pick = next((c for c in cands if c.branch.id == choice.branch_id), cands[0])

            tries = 0
            blocked: dict[str, str] = {}
            while pick is not None and tries < cfg.max_branch_tries:
                tries += 1
                budget.spend("get_chunks", pick.branch.id)
                chunks = self._safe_chunks(pick.branch.id, groups, trace, question)
                siblings = [(c.branch, self._safe_chunks(c.branch.id, groups, [], question))
                            for c in cands[:3] if c.branch.id != pick.branch.id and c.branch.id not in blocked]
                budget.spend("evidence_check", pick.branch.id)
                check = self.llm.check(q, pick.branch, chunks, siblings, df, n_docs, store.branches.get(pick.branch.parent_id))
                log(event="evidence check", branch=pick.branch.id, ok=check.ok, reason=check.reason,
                    ratio=round(check.ratio, 2), quotes=check.supports[:3])
                if check.ok:
                    return self._respond(question, q, pick.branch, chunks, session, trace, budget, ids, tries, False)
                blocked[pick.branch.id] = check.reason          # soft block: this question only, kept in the list
                log(event="soft-block", branch=pick.branch.id, reason=check.reason)
                pick = next((c for c in cands if c.branch.id not in blocked and c.score >= cfg.score_floor), None)

            log(event="tries exhausted or no branch left", blocked=list(blocked))
            return self._fallback(q, question, groups, session, trace, budget, ids, tries)
        except (BudgetExceeded, LoopDetected) as exc:
            log(event="guardrail", detail=str(exc))
            return Result(question, "refused", f"I couldn't finish within my limits ({exc}). Please rephrase or narrow the question.")

    # ---- ranking (plan 6.3): path-aware leaf scoring --------------------------------------------------------------------------
    def _rank(self, q: str, groups) -> list[Candidate]:
        qvec = self.store.embedder.embed(q)
        topic_scores = {t.id: cosine(qvec, t.embedding) for t in self.store.topics(groups)}
        w = self.config.topic_weight
        out = []
        for leaf in self.store.leaves(groups):
            ts, cs = topic_scores.get(leaf.parent_id, 0.0), cosine(qvec, leaf.embedding)
            out.append(Candidate(leaf, w * ts + (1 - w) * cs, ts, cs))
        out.sort(key=lambda c: c.score, reverse=True)
        return out[:self.config.top_k_branches]

    def _topic_only(self, q: str, top: Candidate) -> bool:
        """True when every word in the question is already covered by the topic itself, so no word
        points at one particular type (e.g. just "refund" or "billing")."""
        topic = self.store.branches.get(top.branch.parent_id)
        if topic is None:
            return False
        words = set(content_stems(q))
        if not words:
            return False
        siblings = [b for b in self.store.branches.values() if b.parent_id == topic.id and b.status == "active"]
        if len(siblings) <= 1:
            return False                                    # nothing to choose between
        topic_words = set(content_stems(topic.search_text()))
        topic_name = set(content_stems(topic.name))
        leaf = top.branch
        leaf_specific = (set(content_stems(leaf.name)) - topic_name) | (set(content_stems(leaf.search_text())) - topic_words)
        return words <= topic_words and not (words & leaf_specific)

    def _safe_chunks(self, branch_id: str, groups, trace: list, question: str) -> list[Chunk]:
        """Fetch chunks and strip anything that tries to give the AI orders (plan section 8)."""
        out = []
        for c in self.store.chunks_for(branch_id, groups)[:self.config.chunks_to_llm]:
            clean, removed = sanitize(c.text)
            if removed:
                trace.append({"q": question, "event": "prompt-injection removed", "chunk": c.id, "removed": removed})
            out.append(Chunk(**{**c.__dict__, "text": clean}))
        return out

    # ---- fallback path (plan 6.5) --------------------------------------------------------------------------------------------
    def _fallback(self, q, question, groups, session, trace, budget, ids, tries) -> Result:
        cfg = self.config
        raw_words, df, n_docs = self.store.vocab(groups)
        try:
            budget.spend("search_all_chunks", q)
            qvec = self.store.embedder.embed(q)
            qs = set(content_stems(q))
            scored = []
            for c, cos in self.store.search_chunks(qvec, groups, 50):
                overlap = len(qs & set(content_stems(c.text))) / (len(qs) or 1)
                scored.append((cos + cfg.keyword_weight * overlap, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = [(s, c) for s, c in scored[:cfg.fallback_top_k] if s >= cfg.fallback_floor]
            trace.append({"q": question, "event": "fallback direct search",
                          "top": [(c.id, round(s, 3)) for s, c in top[:3]]})
            if top:
                branch = self.store.branches[top[0][1].branch_id]
                chunks = self._safe_chunks(branch.id, groups, trace, question)
                budget.spend("evidence_check", "fallback:" + branch.id)
                check = self.llm.check(q, branch, chunks, [], df, n_docs, self.store.branches.get(branch.parent_id))
                trace.append({"q": question, "event": "fallback evidence check", "branch": branch.id,
                              "ok": check.ok, "reason": check.reason})
                if check.ok:
                    return self._respond(question, q, branch, chunks, session, trace, budget, ids, tries, True)
        except (BudgetExceeded, LoopDetected) as exc:
            trace.append({"q": question, "event": "guardrail", "detail": str(exc)})
        names = ", ".join(self.store.branches[i].name for i in ids) or "none"
        return Result(question, "refused",
                      "I couldn't find a reliable answer in the knowledge base. "
                      f"Closest topics I checked: {names}. Could you rephrase or add more detail?",
                      candidate_ids=ids, tries=tries, used_fallback=True)

    # ---- answer + faithfulness (plan 6.8) -----------------------------------------------------------------------------------------
    def _respond(self, question, q, branch, chunks, session, trace, budget, ids, tries, used_fallback) -> Result:
        budget.spend("answer", branch.id)
        body = self.llm.answer(q, branch, chunks)
        bad = self.llm.unsupported(body, chunks)
        if bad:
            trace.append({"q": question, "event": "faithfulness: removed", "sentences": bad})
            body = " ".join(s for s in re.split(r"(?<=[.!?])\s+", body) if s not in bad)
        if not body.strip():
            return Result(question, "refused", "I found related text but could not support an answer from it.",
                          branch_id=branch.id, candidate_ids=ids, tries=tries, used_fallback=used_fallback)
        calc = try_calculate(question)
        if calc:
            budget.spend("calculator", calc[0])
            body += f"\n\nCalculation: {calc[0]} = {calc[1]:.2f}"
            trace.append({"q": question, "event": "calculator", "expression": calc[0], "result": calc[1]})
        sources = sorted({c.source for c in chunks if c.source})
        cite = f"{branch.path}" + (f" ({', '.join(sources)})" if sources else "")
        session["last_topic"] = self.store.branches[branch.parent_id].name if branch.parent_id else branch.name
        return Result(question, "answered", f"{body}\n\nSource: {cite}", branch_id=branch.id, citations=[cite],
                      candidate_ids=ids, tries=tries, used_fallback=used_fallback)
