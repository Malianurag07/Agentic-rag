import json
import re
import tempfile
import unittest
from pathlib import Path

import docx

from agentic_rag.blocks import Block
from agentic_rag.chunker import chunk_subsection
from agentic_rag.config import Config
from agentic_rag.ingest_docs import ingest_files
from agentic_rag.loaders import load_document, load_txt
from agentic_rag.sqlite_store import SqliteStore
from agentic_rag.structure import build_tree

ROOT = Path(__file__).resolve().parent.parent
REAL = {"txt": ROOT / "enterprise_company_policy_manual.txt", "docx": ROOT / "enterprise_company_policy_manual.docx",
        "pdf": ROOT / "Rag testing doc.pdf"}

SAMPLE_TXT = """SECTION 1: GENERAL RULES

1.1 Scope (Narrative): This policy applies to everyone. It is binding.

1.2 LIMITS TABLE (STRUCTURED)

  [TABLE COLUMNS] Tier  |  Limit  |  Approver
  -----------------------------------------
  * Tier 1  --  $100  --  Manager
  * Tier 2  --  $5,000  --  Director

MANDATORY CROSS-DOMAIN RULE [R-02 | NO RETALIATION]: Retaliation is prohibited.

SECTION 2: OTHER

2.1 Misc: Something else entirely.
"""


class TxtAndStructure(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "policy.txt"
        self.path.write_text(SAMPLE_TXT)

    def tearDown(self):
        self.tmp.cleanup()

    def test_txt_table_is_parsed_as_a_table(self):
        tables = [b for b in load_txt(self.path) if b.kind == "table"]
        self.assertEqual(tables[0].header, ["Tier", "Limit", "Approver"])
        self.assertEqual(tables[0].rows[1], ["Tier 2", "$5,000", "Director"])

    def test_tree_has_sections_and_subsections(self):
        tree = build_tree(load_txt(self.path))
        # the sample has no text before "SECTION 1", so there is no front-matter section
        self.assertEqual([s.number for s in tree], ["1", "2"])
        self.assertEqual([s.number for s in tree[0].subsections], ["1.1", "1.2"])
        self.assertEqual(tree[0].subsections[0].title, "Scope (Narrative)")

    def test_numbered_paragraph_in_the_wrong_section_is_not_a_heading(self):
        blocks = [Block("para", text="SECTION 1: A"), Block("para", text="1.1 Title: body"),
                  Block("para", text="7.3 Something: not a real subsection of section 1")]
        subs = build_tree(blocks)[0].subsections
        self.assertEqual([s.number for s in subs], ["1.1"])

    def test_chunking_isolates_rules_and_repeats_table_headers(self):
        tree = build_tree(load_txt(self.path))
        chunks = [c for sub in tree[0].subsections for c in chunk_subsection(sub, Config())]
        rule = [c for c in chunks if c.kind == "rule"]
        self.assertEqual(rule[0].meta["rule_ids"], ["R-02"])
        table = [c for c in chunks if c.kind == "table"]
        self.assertTrue(all("Columns: Tier | Limit | Approver" in c.text for c in table))
        self.assertIn("Limit: $5,000", table[0].text)

    def test_long_table_is_split_and_every_piece_keeps_its_columns(self):
        rows = [[f"Row {i}", "word " * 30] for i in range(12)]
        sub = build_tree([Block("para", text="SECTION 1: A"), Block("table", header=["Name", "Text"], rows=rows)])[0].subsections[0]
        pieces = chunk_subsection(sub, Config(table_chunk_words=80))
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all("Columns: Name | Text" in c.text for c in pieces))

    def test_over_long_paragraph_is_cut_on_sentence_boundaries(self):
        text = " ".join(f"This is sentence number {i} of a very long paragraph." for i in range(80))
        sub = build_tree([Block("para", text="SECTION 1: A"), Block("para", text=text)])[0].subsections[0]
        pieces = chunk_subsection(sub, Config(doc_chunk_words=60))
        self.assertGreater(len(pieces), 3)
        self.assertTrue(all(len(c.text.split()) <= 70 and c.text.rstrip().endswith(".") for c in pieces))

    def test_figures_are_flagged_not_dropped(self):
        sub = build_tree([Block("para", text="SECTION 1: A"), Block("para", text="Figure 1.1: Chart of things."),
                          Block("figure", text="[image]", page=2)])[0].subsections[0]
        figs = [c for c in chunk_subsection(sub, Config()) if c.kind == "figure"]
        self.assertTrue(figs[0].needs_vision)
        self.assertIn("Figure 1.1: Chart of things.", figs[0].text)


class DocxLoader(unittest.TestCase):
    def test_docx_paragraphs_tables_and_bullets_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.docx"
            d = docx.Document()
            d.add_paragraph("SECTION 1: TEST")
            t = d.add_table(rows=2, cols=2)
            t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Col A", "Col B"
            t.rows[1].cells[0].text, t.rows[1].cells[1].text = "a1", "b1"
            d.add_paragraph("after the table", style="List Bullet")
            d.save(path)
            blocks = load_document(path)
            self.assertEqual([b.kind for b in blocks], ["para", "table", "para"])
            self.assertEqual(blocks[1].rows, [["a1", "b1"]])
            self.assertTrue(blocks[2].text.startswith("•"))

    def test_legacy_doc_gives_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "convert it to .docx"):
            load_document(Path("old.doc"))


class SqliteLayer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.txt = Path(self.tmp.name) / "policy.txt"
        self.txt.write_text(SAMPLE_TXT)
        self.db = Path(self.tmp.name) / "docs.db"
        self.logs: list[str] = []

    def tearDown(self):
        self.tmp.cleanup()

    def ingest(self):
        store = ingest_files([self.txt], self.db, log=self.logs.append)
        self.addCleanup(store.close)
        return store

    def test_rows_embeddings_and_keyword_index_are_written(self):
        store = self.ingest()
        n = store.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        self.assertGreater(n, 3)
        self.assertEqual(store.db.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL").fetchone()[0], 0)
        self.assertEqual(store.db.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0], n)
        self.assertTrue(store.keyword_search("retaliation"))
        self.assertEqual(store.db.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_unchanged_file_is_skipped_and_changed_file_replaces_old_rows(self):
        self.ingest()
        self.logs.clear()
        self.ingest()
        self.assertTrue(any("skip" in l for l in self.logs))
        self.txt.write_text(SAMPLE_TXT.replace("Retaliation is prohibited.", "Reprisals are forbidden."))
        store = self.ingest()
        self.assertFalse(store.keyword_search("prohibited"))      # the old text is gone
        self.assertTrue(store.keyword_search("forbidden"))        # the new text is searchable
        self.assertEqual(store.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 1)

    def test_pipeline_store_can_be_rebuilt_from_sqlite(self):
        mem = self.ingest().load_store()
        self.assertTrue(any(b.level == 2 for b in mem.branches.values()))
        self.assertTrue(all(c.embedding for c in mem.chunks))

    def test_keyword_search_survives_punctuation(self):
        store = self.ingest()
        self.assertIsInstance(store.keyword_search('"; DROP TABLE chunks; --'), list)


@unittest.skipUnless(all(p.exists() for p in REAL.values()), "the uploaded sample documents are not present")
class UploadedDocuments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocks = {k: load_document(p) for k, p in REAL.items()}
        cls.trees = {k: build_tree(b) for k, b in cls.blocks.items()}

    def test_every_format_yields_the_same_structure(self):
        for fmt, tree in self.trees.items():
            subs = [s for sec in tree for s in sec.subsections]
            self.assertEqual(len(tree), 19, fmt)                 # front matter + 18 sections
            self.assertEqual(len(subs), 65, fmt)

    def test_every_format_finds_all_thirty_tables(self):
        for fmt, blocks in self.blocks.items():
            self.assertEqual(sum(b.kind == "table" for b in blocks), 30, fmt)

    def test_pdf_tables_match_docx_tables_cell_for_cell(self):
        norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
        cells = lambda t: [norm(c) for r in [t.header] + t.rows for c in r]
        docx_t = [b for b in self.blocks["docx"] if b.kind == "table"]
        pdf_t = [b for b in self.blocks["pdf"] if b.kind == "table"]
        self.assertEqual([cells(t) for t in docx_t], [cells(t) for t in pdf_t])

    def test_currency_amounts_survive_pdf_extraction(self):
        text = " ".join(c for b in self.blocks["pdf"] if b.kind == "table" for r in b.rows for c in r)
        self.assertIn("$0.01 – $4,999", text)

    def test_page_headers_and_footers_are_removed_from_the_pdf(self):
        text = " ".join(b.text for b in self.blocks["pdf"] if b.kind == "para")
        self.assertNotIn("Page 3 of 30", text)
        self.assertNotIn("CLASSIFICATION: INTERNAL GOVERNANCE STANDARD", text)

    def test_seventeen_rule_chunks_with_the_expected_ids(self):
        for fmt, tree in self.trees.items():
            ids = [i for sec in tree for sub in sec.subsections for c in chunk_subsection(sub, Config())
                   if c.kind == "rule" for i in c.meta["rule_ids"]]
            self.assertEqual(sorted(ids), ["R-01"] * 4 + ["R-02"] * 4 + ["R-03"] * 3 + ["R-04"] * 3 + ["R-05"] * 3, fmt)

    def test_charts_are_kept_as_flagged_figures_in_pdf_and_docx(self):
        for fmt in ("pdf", "docx"):
            figs = [c for sec in self.trees[fmt] for sub in sec.subsections for c in chunk_subsection(sub, Config()) if c.kind == "figure"]
            self.assertEqual(len(figs), 2, fmt)
            self.assertTrue(all(f.needs_vision for f in figs))

    def test_stored_chunks_agree_across_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ingest_files(list(REAL.values()), Path(tmp) / "all.db", log=lambda *_: None)
            agreement = store.stats()["agreement"]
            store.close()
            self.assertTrue(all(v >= 0.85 for v in agreement.values()), agreement)


if __name__ == "__main__":
    unittest.main()
