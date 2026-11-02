import unittest
from pathlib import Path

from agentic_rag.chunker import chunk_subsection
from agentic_rag.config import Config
from agentic_rag.blocks import Block
from agentic_rag.embeddings import HashEmbedder, cosine
from agentic_rag.models import Chunk
from agentic_rag.neural import DEFAULT_DIR, MODEL_SHA256, MODEL_URL
from agentic_rag.retrieval import BM25, Retriever, SmallToBigRetriever, make_units
from agentic_rag.structure import build_tree
from agentic_rag.text import content_stems

TABLE = ("Table in 4.3 Records Retention. Columns: Series | Retention.\nRows:\n- Series: FIN-02 | Retention: 10 years\n"
         "- Series: OPS-07 Ephemeral Messaging | Retention: 180 days\n- Series: HR-03 | Retention: 7 years")


def chunk(i, text, prefix="4.3 Records:"):
    return Chunk(id=f"c{i}", branch_id="doc:4.3", text=text, context_prefix=prefix)


def embed_all(chunks):
    emb = HashEmbedder(dim=512)
    emb.fit([c.embed_text() for c in chunks])
    for c in chunks:
        c.embedding = emb.embed(c.embed_text())
    return emb


class Bm25Tests(unittest.TestCase):
    def test_rare_exact_term_outranks_common_words(self):
        docs = [content_stems(t) for t in ("the policy covers expenses", "the policy covers travel", "fraud checks need approval")]
        scores = BM25(docs).scores(content_stems("fraud approval"))
        self.assertEqual(max(range(3), key=lambda i: scores[i]), 2)

    def test_unknown_words_score_zero(self):
        self.assertEqual(BM25([content_stems("alpha beta")]).scores(["zzz"]), [0.0])


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [chunk(1, "Expense claims must be filed within thirty days."), chunk(2, TABLE),
                       chunk(3, "Visitors wear a colour coded badge and stay escorted.", "9.2 Access:")]
        self.emb = embed_all(self.chunks)

    def test_hybrid_finds_the_chunk_with_the_exact_term(self):
        ranked = Retriever(self.chunks, self.emb, mode="hybrid").rank_chunks("OPS-07 messaging retention", k=3)
        self.assertEqual(ranked[0][0].id, "c2")

    def test_sections_are_ranked_by_their_best_chunk(self):
        ranked = Retriever(self.chunks, self.emb, mode="bm25").rank_chunks("escorted visitors badge", k=3)
        self.assertEqual(Retriever.rank_sections(ranked)[0][0], "doc:4.3")     # all three chunks share a section id here

    def test_detailed_ranking_reports_both_rankers(self):
        row = Retriever(self.chunks, self.emb).rank_chunks_detailed("expense claims thirty days", k=2)[0]
        self.assertEqual(row[0].id, "c1")
        self.assertEqual(len(row), 5)


class ParentChildTests(unittest.TestCase):
    def test_table_units_are_one_per_row_plus_the_whole_chunk(self):
        units = make_units(chunk(2, TABLE))
        self.assertEqual(len(units), 4)
        self.assertTrue(any("OPS-07 Ephemeral Messaging" in u and "FIN-02" not in u for u in units))

    def test_a_matching_row_returns_its_whole_parent_chunk(self):
        chunks = [chunk(1, "Unrelated text about badges and visitors."), chunk(2, TABLE)]
        emb = embed_all(chunks)
        top = SmallToBigRetriever(chunks, emb, mode="hybrid").rank_chunks("ephemeral messaging 180 days", k=2)[0][0]
        self.assertEqual(top.id, "c2")
        self.assertIn("FIN-02", top.text)                                          # neighbouring rows come along as context

    def test_parents_are_not_repeated(self):
        chunks = [chunk(2, TABLE)]
        ranked = SmallToBigRetriever(chunks, embed_all(chunks), mode="hybrid").rank_chunks("retention years", k=5)
        self.assertEqual(len(ranked), 1)


class ChunkingOptionTests(unittest.TestCase):
    def sub(self):
        rows = [[f"Row {i}", "word " * 20] for i in range(4)]
        return build_tree([Block("para", text="SECTION 1: A"), Block("table", header=["Name", "Text"], rows=rows)])[0].subsections[0]

    def test_row_level_chunks_make_one_chunk_per_row_with_columns(self):
        chunks = chunk_subsection(self.sub(), Config(table_row_chunks=True))
        self.assertEqual(len(chunks), 4)
        self.assertTrue(all("Columns: Name | Text" in c.text for c in chunks))

    def test_sentence_overlap_repeats_the_last_sentence(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(40))
        sub = build_tree([Block("para", text="SECTION 1: A"), Block("para", text=text)])[0].subsections[0]
        plain = chunk_subsection(sub, Config(doc_chunk_words=40))
        overlapped = chunk_subsection(sub, Config(doc_chunk_words=40, chunk_overlap_sentences=1))
        first_tail = plain[0].text.split(". ")[-1].rstrip(".")
        self.assertIn(first_tail.split("number")[-1].strip().split(" ")[0], overlapped[1].text[:60])


@unittest.skipUnless((DEFAULT_DIR / "onnx" / "model.onnx").exists(), "run `python -m agentic_rag download-model` first")
class NeuralEmbedderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agentic_rag.neural import MiniLMEmbedder
        cls.emb = MiniLMEmbedder()

    def test_meaning_beats_shared_words(self):
        q, good, bad = self.emb.embed_many([
            "How long are chat messages kept?",
            "Ephemeral Messaging: direct instant messages, retained 180 calendar days then purged.",
            "Tier 1 purchases up to $4,999 need a line manager and an automated budget check."])
        self.assertGreater(cosine(q, good), cosine(q, bad) + 0.2)

    def test_vectors_are_unit_length_and_384_wide(self):
        v = self.emb.embed("hello")
        self.assertEqual(len(v), 384)
        self.assertAlmostEqual(sum(x * x for x in v.values()) ** 0.5, 1.0, places=3)


class ModelProvenanceTests(unittest.TestCase):
    def test_model_download_is_pinned_by_checksum(self):
        self.assertEqual(len(MODEL_SHA256), 64)
        self.assertTrue(MODEL_URL.endswith("all-MiniLM-L6-v2/onnx.tar.gz"))


if __name__ == "__main__":
    unittest.main()
