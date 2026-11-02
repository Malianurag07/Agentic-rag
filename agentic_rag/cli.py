import argparse
import json
from pathlib import Path

from .config import Config
from .evaluate import format_report, load_testset, run_eval
from .ingest import build_store
from .llm import make_llm
from .pipeline import Pipeline
from .store import Store


def _pipeline(args) -> Pipeline:
    config = Config()
    return Pipeline(Store.load(args.index), make_llm(args.llm, config), config)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="agentic_rag", description="Agentic RAG over a hierarchical knowledge base")
    ap.add_argument("--index", default="data/index.json")
    ap.add_argument("--llm", choices=["heuristic", "claude"], default="heuristic",
                    help="heuristic = offline rules (default); claude = Claude via the Anthropic SDK")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="build the index from Markdown files")
    p.add_argument("--kb", default="data/kb")

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question")
    p.add_argument("--groups", default="public", help="comma separated access groups")
    p.add_argument("--trace", action="store_true")

    sub.add_parser("chat", help="interactive session with follow-up memory")

    p = sub.add_parser("eval", help="run the test set")
    p.add_argument("--testset", default="data/testset.jsonl")
    p.add_argument("--out", default="reports/latest.json")

    args = ap.parse_args(argv)
    if args.cmd == "ingest":
        store = build_store(args.kb)
        store.save(args.index)
        print(f"Indexed {len(store.branches)} branches and {len(store.chunks)} chunks -> {args.index}")
    elif args.cmd == "ask":
        res = _pipeline(args).ask(args.question, args.groups.split(","))
        print(res.answer)
        print(f"\n[{res.status}] branch={res.branch_id} steps={res.steps} judgment_calls={res.llm_calls} "
              f"tries={res.tries} fallback={res.used_fallback}")
        if args.trace:
            print(json.dumps(res.trace, indent=2, default=str))
    elif args.cmd == "chat":
        pipe, session = _pipeline(args), {}
        print("Ask a question (empty line to quit).")
        while (q := input("> ").strip()):
            print(pipe.ask(q, ["public"], session).answer, "\n")
    elif args.cmd == "eval":
        report = run_eval(_pipeline(args), load_testset(args.testset))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(format_report(report))
        print(f"\nFull report: {args.out}")
