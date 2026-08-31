from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from tools.eagle3_corpus import build_corpus
from tools.eagle3_corpus import select_eval_subset


class EvalSubsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        tool_dir = Path(build_corpus.__file__).resolve().parent
        self.corpus = self.root / "corpus"
        build_corpus.build(
            catalog_path=tool_dir / "catalog.json",
            policy_path=tool_dir / "leakage_rules.json",
            schema_path=tool_dir / "corpus_job.schema.json",
            output=self.corpus,
            profile="standard",
            forbidden_path=None,
            expected_forbidden_sha256=None,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_selects_reproducible_stratified_hundred(self) -> None:
        first = self.root / "first"
        second = self.root / "second"
        manifest = select_eval_subset.build_subset(
            corpus_dir=self.corpus, output=first, count=100
        )
        select_eval_subset.build_subset(corpus_dir=self.corpus, output=second, count=100)
        self.assertEqual((first / "eval.jobs.jsonl").read_bytes(), (second / "eval.jobs.jsonl").read_bytes())
        self.assertEqual((first / "manifest.json").read_bytes(), (second / "manifest.json").read_bytes())
        records = build_corpus.read_jsonl(first / "eval.jobs.jsonl")
        self.assertEqual(len(records), 100)
        self.assertEqual(
            Counter(record["category"] for record in records),
            Counter(
                {
                    "technical_instructional": 45,
                    "code_system_design": 25,
                    "general": 30,
                }
            ),
        )
        self.assertEqual(len({record["source"]["family_id"] for record in records}), 7)
        self.assertTrue(manifest["selection"]["source_order_preserved"])

    def test_rejects_tampered_source_eval(self) -> None:
        with (self.corpus / "eval.jobs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("{}\n")
        with self.assertRaises(build_corpus.CorpusError):
            select_eval_subset.build_subset(
                corpus_dir=self.corpus, output=self.root / "subset", count=100
            )

    def test_rejects_oversized_subset(self) -> None:
        with self.assertRaises(build_corpus.CorpusError):
            select_eval_subset.build_subset(
                corpus_dir=self.corpus, output=self.root / "subset", count=401
            )


if __name__ == "__main__":
    unittest.main()
