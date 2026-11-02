"""Evaluate the pipeline on the document question set (data/doc_questions.jsonl) against the SQLite-stored documents.

Every question is asked of each document (PDF, DOCX, TXT) separately, because they are the same manual in three formats and
should give the same answers. Judging is by explicit rules, with no LLM involved:
    status   did it answer / refuse / ask a question as expected?
    facts    does the answer contain every expected fact? (multi-part: how many of them)
    section  did the pipeline pick a subsection that holds the answer?"""

import json
import re
import time
from collections import defaultdict
from pathlib import Path

from .config import Config
from .llm import make_llm
from .pipeline import Pipeline
from .sqlite_store import SqliteStore
from .store import Store

NORM = lambda s: re.sub(r"\s+", " ", s.lower().replace("‑", "-")).strip()


def only_document(store: Store, doc_id: str) -> Store:
    """A view of the store containing a single document, so formats are tested separately."""
    prefix = doc_id + ":"
    branches = [b for b in store.branches.values() if b.id.startswith(prefix)]
    chunks = [c for c in store.chunks if c.id.startswith(prefix)]
    return Store(store.embedder, branches, chunks)


def load_questions(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _section_of(branch_id: str | None) -> str | None:
    return branch_id.split(":", 1)[1] if branch_id and ":" in branch_id else None


def judge(q: dict, res) -> dict:
    want = q["status"] if isinstance(q["status"], list) else [q["status"]]
    got_sections = {_section_of(b) for b in (res.branch_ids or [res.branch_id]) if b}
    cand_sections = {_section_of(c) for c in res.candidate_ids}
    out = {"status_ok": res.status in want}
    answer = NORM(res.answer)
    facts = [NORM(f) for f in q.get("contains", [])]
    out["facts_found"], out["facts_total"] = sum(f in answer for f in facts), len(facts)
    out["facts_ok"] = out["facts_found"] == out["facts_total"]
    reqs = q.get("sections", [])
    sat = lambda pool: all(any(s in pool for s in (need if isinstance(need, list) else [need])) for need in reqs)
    out["section_ok"] = sat(got_sections) if reqs else None
    out["section_in_top3"] = sat(got_sections | cand_sections) if reqs else None
    if "refused" in want and want == ["refused"]:
        out["passed"] = out["status_ok"]
        out["hallucinated"] = res.status == "answered"
    else:
        out["passed"] = out["status_ok"] and (out["facts_ok"] if res.status == "answered" and facts else True)
        out["hallucinated"] = False
    return out


def run(db_path: str | Path, questions_path: str | Path, llm_kind: str = "heuristic", config: Config | None = None) -> dict:
    config = config or Config()
    sqlite = SqliteStore(db_path)
    full = sqlite.load_store()
    docs = [dict(r) for r in sqlite.db.execute("SELECT id, format FROM documents ORDER BY format")]
    questions = load_questions(questions_path)
    rows = []
    for d in docs:
        pipe = Pipeline(only_document(full, d["id"]), make_llm(llm_kind, config), config)
        for q in questions:
            session: dict = {}
            for earlier in q.get("history", []):
                pipe.ask(earlier, ["public"], session)
            res = pipe.ask(q["question"], ["public"], session)
            j = judge(q, res)
            rows.append({"format": d["format"], "id": q["id"], "kind": q["kind"], "question": q["question"], "status": res.status,
                         "branch": _section_of(res.branch_id), "expected_sections": q.get("sections"), "answer": res.answer,
                         "steps": res.steps, "llm_calls": res.llm_calls, "fallback": res.used_fallback,
                         "seconds": round(res.elapsed, 4), **j})
    return {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "llm": llm_kind, "rows": rows, "summary": summarize(rows)}


def _pct(a: int, b: int) -> float:
    return a / b if b else 0.0


def summarize(rows: list[dict]) -> dict:
    s: dict = {"runs": len(rows), "passed": sum(r["passed"] for r in rows)}
    s["pass_rate"] = _pct(s["passed"], s["runs"])
    answerable = [r for r in rows if r["facts_total"] or r["section_ok"] is not None]
    s["section_top1"] = _pct(sum(bool(r["section_ok"]) for r in answerable), len(answerable))
    s["section_top3"] = _pct(sum(bool(r["section_in_top3"]) for r in answerable), len(answerable))
    s["fact_recall"] = _pct(sum(r["facts_found"] for r in rows), sum(r["facts_total"] for r in rows))
    unans = [r for r in rows if r["kind"] in ("unanswerable", "trap", "chart-only")]
    s["refusal_rate"] = _pct(sum(r["status"] == "refused" for r in unans), len(unans))
    s["hallucinations"] = sum(r["hallucinated"] for r in rows)
    wrong = [r for r in rows if r["status"] == "answered" and r["facts_total"] and not r["facts_ok"]]
    s["confident_wrong"] = len(wrong)
    s["refused_answerable"] = sum(r["status"] == "refused" and r["facts_total"] > 0 for r in rows)
    s["avg_steps"] = sum(r["steps"] for r in rows) / len(rows)
    s["avg_seconds"] = sum(r["seconds"] for r in rows) / len(rows)
    by_kind, by_fmt = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for r in rows:
        by_kind[r["kind"]][0] += r["passed"]; by_kind[r["kind"]][1] += 1
        by_fmt[r["format"]][0] += r["passed"]; by_fmt[r["format"]][1] += 1
    s["by_kind"], s["by_format"] = dict(by_kind), dict(by_fmt)
    return s


def format_report(report: dict, show_failures: int = 40) -> str:
    s = report["summary"]
    pc = lambda x: f"{x * 100:5.1f}%"
    lines = [f"Document question evaluation ({report['timestamp']}, judgment layer: {report['llm']})", "=" * 78,
             f"Runs: {s["runs"]} ({s["runs"] // 3} questions x 3 formats)   Passed: {s['passed']}   Pass rate: {pc(s['pass_rate'])}", "",
             f"Right section chosen (top-1): {pc(s['section_top1'])}    in top 3 candidates: {pc(s['section_top3'])}",
             f"Expected facts present in answers: {pc(s['fact_recall'])}",
             f"Unanswerable questions correctly refused: {pc(s['refusal_rate'])}   invented answers: {s['hallucinations']}",
             f"Answered with the wrong/incomplete facts: {s['confident_wrong']}   refused although answerable: {s['refused_answerable']}",
             f"Avg agent steps {s['avg_steps']:.1f}   avg time {s['avg_seconds'] * 1000:.0f} ms per question", "", "By format:"]
    for k, (a, b) in sorted(s["by_format"].items()):
        lines.append(f"  {k:<6}{a}/{b}  {pc(a / b)}")
    lines += ["", "By question type:"]
    for k, (a, b) in s["by_kind"].items():
        lines.append(f"  {k:<14}{a:>3}/{b:<3} {pc(a / b)}")
    failed = [r for r in report["rows"] if not r["passed"] and r["format"] == "txt"]
    lines += ["", f"Failures on the TXT version ({len(failed)} of {s['runs'] // 3}; the other formats are in the JSON report):"]
    for r in failed[:show_failures]:
        lines.append(f"  [{r['id']}] {r['question'][:84]}")
        lines.append(f"       status={r['status']} section={r['branch']} (want {r['expected_sections']}) facts {r['facts_found']}/{r['facts_total']}")
        lines.append(f"       -> {r['answer'][:150].replace(chr(10), ' ')}")
    return "\n".join(lines)
