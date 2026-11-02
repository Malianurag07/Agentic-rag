"""Evaluation runner (plan section 9): a test set in, per-question results and aggregate metrics out."""

import json
import time
from collections import defaultdict
from pathlib import Path

from .pipeline import Pipeline


def load_testset(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def judge(case: dict, res) -> dict:
    """Score one result against its expectations."""
    want_status = case.get("status", "answered")
    checks = {}
    if want_status != "any":                       # "any" = only the text checks below matter (e.g. a permission test)
        checks["status"] = res.status == want_status
    if "branch" in case:
        checks["branch_top1"] = res.branch_id == case["branch"]
        checks["branch_top3"] = case["branch"] in res.candidate_ids or checks["branch_top1"]
    if "contains" in case:
        checks["contains"] = all(s.lower() in res.answer.lower() for s in case["contains"])
    if "not_contains" in case:
        checks["not_contains"] = not any(s.lower() in res.answer.lower() for s in case["not_contains"])
    return checks


def run_eval(pipeline: Pipeline, cases: list[dict]) -> dict:
    rows = []
    for case in cases:
        session: dict = {}
        groups = case.get("groups", ["public"])
        for earlier in case.get("history", []):
            pipeline.ask(earlier, groups, session)
        res = pipeline.ask(case["question"], groups, session)
        checks = judge(case, res)
        rows.append({"id": case["id"], "kind": case["kind"], "question": case["question"], "passed": all(checks.values()),
                     "checks": checks, "status": res.status, "branch": res.branch_id, "expected_branch": case.get("branch"),
                     "answer": res.answer, "steps": res.steps, "llm_calls": res.llm_calls, "tries": res.tries,
                     "fallback": res.used_fallback, "seconds": round(res.elapsed, 4)})
    return {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "llm": pipeline.llm.name,
            "embedder": pipeline.store.embedder.kind, "summary": summarize(rows), "rows": rows}


def _rate(rows, key):
    vals = [r["checks"][key] for r in rows if key in r["checks"]]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def summarize(rows: list[dict]) -> dict:
    out = {"questions": len(rows), "passed": sum(r["passed"] for r in rows)}
    out["pass_rate"] = out["passed"] / len(rows) if rows else 0
    for key in ("branch_top1", "branch_top3", "contains", "not_contains"):
        out[key], out[key + "_n"] = _rate(rows, key)
    out["avg_steps"] = sum(r["steps"] for r in rows) / len(rows)
    out["avg_llm_calls"] = sum(r["llm_calls"] for r in rows) / len(rows)
    out["fallback_rate"] = sum(r["fallback"] for r in rows) / len(rows)
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r["passed"])
    out["by_kind"] = {k: (sum(v), len(v)) for k, v in by_kind.items()}
    return out


def format_report(report: dict) -> str:
    s = report["summary"]
    pct = lambda x: "n/a" if x is None else f"{x * 100:5.1f}%"
    lines = [f"Agentic RAG evaluation  ({report['timestamp']}, llm={report['llm']}, embedder={report['embedder']})",
             "-" * 78,
             f"Questions: {s['questions']}   Passed: {s['passed']}   Overall pass rate: {pct(s['pass_rate'])}",
             f"Branch top-1: {pct(s['branch_top1'])} (n={s['branch_top1_n']})   Branch top-3: {pct(s['branch_top3'])}",
             f"Answer contains expected facts: {pct(s['contains'])} (n={s['contains_n']})   "
             f"No forbidden text: {pct(s['not_contains'])} (n={s['not_contains_n']})",
             f"Avg agent steps: {s['avg_steps']:.1f}   Avg judgment calls: {s['avg_llm_calls']:.1f}   "
             f"Fallback used: {pct(s['fallback_rate'])}", "", "By question type:"]
    for kind, (ok, n) in sorted(s["by_kind"].items()):
        lines.append(f"  {kind:<18} {ok}/{n}")
    failed = [r for r in report["rows"] if not r["passed"]]
    lines += ["", f"Failures ({len(failed)}):"]
    for r in failed:
        bad = [k for k, v in r["checks"].items() if not v]
        lines.append(f"  [{r['id']}] {r['question']!r}")
        lines.append(f"      failed: {bad} | status={r['status']} branch={r['branch']} (expected {r['expected_branch']})")
        lines.append(f"      answer: {r['answer'][:140].replace(chr(10), ' ')}")
    return "\n".join(lines)
