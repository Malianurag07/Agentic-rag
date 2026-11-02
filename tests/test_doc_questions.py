"""Validates the ANSWER KEY in data/doc_questions.jsonl against the stored documents (not the pipeline).
If one of these fails, the question is wrong or the extraction lost text; it says nothing about retrieval quality."""

import json
import re
import tempfile
import unittest
from pathlib import Path

from agentic_rag.ingest_docs import ingest_files

ROOT = Path(__file__).resolve().parent.parent
FILES = [ROOT / "Rag testing doc.pdf", ROOT / "enterprise_company_policy_manual.docx", ROOT / "enterprise_company_policy_manual.txt"]
QUESTIONS = ROOT / "data" / "doc_questions.jsonl"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("‑", "-")).strip()


def load_questions() -> list[dict]:
    return [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]


@unittest.skipUnless(all(f.exists() for f in FILES) and QUESTIONS.exists(), "documents or question file missing")
class AnswerKey(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.store = ingest_files(FILES, Path(cls.tmp.name) / "q.db", log=lambda *_: None)
        cls.docs = [r["id"] for r in cls.store.db.execute("SELECT id FROM documents")]
        cls.questions = load_questions()

    @classmethod
    def tearDownClass(cls):
        cls.store.close()
        cls.tmp.cleanup()

    def section_text(self, doc: str, number: str) -> str:
        rows = self.store.db.execute("SELECT text FROM chunks WHERE section_id = ?", (f"{doc}:{number}",)).fetchall()
        return norm(" ".join(r["text"] for r in rows))

    def whole_text(self, doc: str) -> str:
        return norm(" ".join(r["text"] for r in self.store.db.execute("SELECT text FROM chunks WHERE document_id = ?", (doc,))))

    def test_there_are_fifty_unique_questions_of_every_kind(self):
        ids = [q["id"] for q in self.questions]
        self.assertEqual(len(ids), 50)
        self.assertEqual(len(set(ids)), 50)
        kinds = {q["kind"] for q in self.questions}
        self.assertTrue({"prose lookup", "table lookup", "multi-part", "golden rule", "paraphrase", "typo", "vague",
                         "follow-up", "unanswerable", "trap", "chart-only"} <= kinds)

    def test_every_expected_fact_is_in_every_format(self):
        """One requirement (possibly with alternatives): all facts must sit together in one of the alternatives.
        Several requirements (a multi-part question): each fact must appear somewhere in the sections named."""
        problems = []
        for q in self.questions:
            reqs = q.get("sections", [])
            facts = [norm(f) for f in q.get("contains", [])]
            for doc in self.docs:
                if len(reqs) == 1:
                    options = reqs[0] if isinstance(reqs[0], list) else [reqs[0]]
                    ok = any(all(f in self.section_text(doc, sec) for f in facts) for sec in options)
                    missing = facts
                else:
                    union = " ".join(self.section_text(doc, sec) for need in reqs for sec in (need if isinstance(need, list) else [need]))
                    missing = [f for f in facts if f not in union]
                    ok = not missing
                if not ok:
                    problems.append((q["id"], doc[-12:], missing))
        self.assertEqual(problems, [])

    def test_every_expected_section_exists_in_every_format(self):
        for q in self.questions:
            for need in q.get("sections", []):
                for s in (need if isinstance(need, list) else [need]):
                    for doc in self.docs:
                        self.assertTrue(self.section_text(doc, s), f"{q['id']}: no text for section {s} in {doc}")

    def test_unanswerable_questions_really_have_no_answer_in_the_text(self):
        absent = {"pet insurance": "pet insurance", "share price": "share price", "name of the current ceo": None,
                  "dress code on fridays": None, "cyber & data domain": None, "tier 5 procurement approval": None}
        for doc in self.docs:
            text = self.whole_text(doc)
            self.assertNotIn("pet insurance", text)
            self.assertNotIn("share price", text)
            self.assertNotIn("stock price", text)
            self.assertNotIn("friday", text)                                    # no dress-code detail exists
            self.assertNotIn("dress code standards", self.section_text(doc, "2.4").replace("workplace violence prevention, consensual relationships & ", "x"))
            self.assertNotRegex(text, r"\b48\b.{0,40}checkpoint|cyber & data.{0,30}\b48\b")      # the chart value is image-only
            self.assertNotRegex(text, r"tier 5.{0,60}\b30[- ]day")

    def test_chart_totals_in_the_text_are_consistent_with_the_chart_questions(self):
        for doc in self.docs:                                                     # the caption states the total, not the split
            self.assertIn("212 auditable control checkpoints", self.whole_text(doc))


if __name__ == "__main__":
    unittest.main()
