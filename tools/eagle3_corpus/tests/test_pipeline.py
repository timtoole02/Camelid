from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from tools.eagle3_corpus import build_corpus as corpus


TOOL_DIR = Path(__file__).resolve().parents[1]
CATALOG_PATH = TOOL_DIR / "catalog.json"
POLICY_PATH = TOOL_DIR / "leakage_rules.json"
SCHEMA_PATH = TOOL_DIR / "corpus_job.schema.json"


class CorpusPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = corpus.load_json(CATALOG_PATH)
        cls.policy = corpus.load_json(POLICY_PATH)
        cls.standard = corpus.all_standard_records(cls.catalog)

    def profile(self, name: str) -> list[dict]:
        return corpus.select_profile(self.standard, self.catalog, name)

    def test_standard_profile_has_exact_counts_and_target_mix(self) -> None:
        records = self.profile("standard")
        audit = corpus.audit_records(
            records,
            catalog=self.catalog,
            policy=self.policy,
            profile="standard",
        )
        self.assertEqual(audit["record_count"], 2800)
        self.assertEqual(audit["counts"]["train"], {
            "technical_instructional": 1080,
            "code_system_design": 600,
            "general": 720,
        })
        self.assertEqual(audit["counts"]["eval"], {
            "technical_instructional": 180,
            "code_system_design": 100,
            "general": 120,
        })
        self.assertEqual(audit["category_ratios"], {
            "technical_instructional": 0.45,
            "code_system_design": 0.25,
            "general": 0.30,
        })

    def test_pilot_is_a_stratified_fifty_record_subset(self) -> None:
        pilot = self.profile("pilot")
        audit = corpus.audit_records(
            pilot,
            catalog=self.catalog,
            policy=self.policy,
            profile="pilot",
        )
        self.assertEqual(len(pilot), 50)
        self.assertEqual(audit["counts"]["train"], {
            "technical_instructional": 18,
            "code_system_design": 10,
            "general": 12,
        })
        self.assertEqual(audit["counts"]["eval"], {
            "technical_instructional": 5,
            "code_system_design": 2,
            "general": 3,
        })
        standard_ids = {record["id"] for record in self.standard}
        self.assertTrue({record["id"] for record in pilot} <= standard_ids)

    def test_every_family_source_and_template_is_preassigned_to_one_split(self) -> None:
        records = self.profile("standard")
        owners: dict[tuple[str, str], set[str]] = {}
        for record in records:
            for kind, value in (
                ("source", record["source"]["source_id"]),
                ("family", record["source"]["family_id"]),
                ("template", record["source"]["template_id"]),
            ):
                owners.setdefault((kind, value), set()).add(record["split"])
        self.assertTrue(owners)
        self.assertTrue(all(len(splits) == 1 for splits in owners.values()))

    def test_duplicate_messages_are_rejected(self) -> None:
        records = self.profile("pilot")
        records.append(deepcopy(records[0]))
        records[-1]["id"] += ".duplicate"
        with self.assertRaisesRegex(corpus.CorpusError, "duplicate message content"):
            corpus.audit_records(
                records,
                catalog=self.catalog,
                policy=self.policy,
                profile="pilot",
            )

    def test_curated_canary_rule_is_rejected(self) -> None:
        records = self.profile("pilot")
        records[0] = deepcopy(records[0])
        records[0]["messages"][1]["content"] += " Include a daily caloric intake summary."
        records[0]["content_sha256"] = corpus.content_digest(records[0]["messages"])
        with self.assertRaisesRegex(corpus.CorpusError, "leakage rule"):
            corpus.audit_records(
                records,
                catalog=self.catalog,
                policy=self.policy,
                profile="pilot",
            )

    def test_exact_protected_span_is_rejected(self) -> None:
        records = self.profile("pilot")
        protected = records[7]["messages"][1]["content"]
        with self.assertRaisesRegex(corpus.CorpusError, "protected exact n-gram"):
            corpus.audit_records(
                records,
                catalog=self.catalog,
                policy=self.policy,
                profile="pilot",
                forbidden_text=protected,
            )

    def test_length_gate_rejects_short_prompt(self) -> None:
        records = self.profile("pilot")
        records[0] = deepcopy(records[0])
        records[0]["messages"][1]["content"] = "Too short."
        records[0]["content_sha256"] = corpus.content_digest(records[0]["messages"])
        with self.assertRaisesRegex(corpus.CorpusError, "user chars outside policy"):
            corpus.audit_records(
                records,
                catalog=self.catalog,
                policy=self.policy,
                profile="pilot",
            )

    def test_records_have_no_assistant_answer_before_exact_target_materialization(self) -> None:
        for record in self.profile("pilot"):
            self.assertEqual([message["role"] for message in record["messages"]], ["system", "user"])
            self.assertEqual(record["generation"], {
                "method": "target_greedy",
                "temperature": 0.0,
                "max_new_tokens": record["generation"]["max_new_tokens"],
            })
            self.assertEqual(record["supervision"]["materialization"], "exact_q4_target_required")

    def test_raw_mask_to_exported_p_plus_two_boundary(self) -> None:
        # Two prompt tokens followed by two exact target-generated assistant tokens.
        input_ids = [101, 102, 201, 202]
        raw_mask = corpus.raw_assistant_loss_mask(
            total_tokens=len(input_ids), assistant_start=2
        )
        self.assertEqual(raw_mask, [0, 1, 1, 0])

        # Final exporter base row P pairs aux[P], embedding(token[P+1]), and a
        # teacher/label for token[P+2].  Its mask must be the raw P+1 row.
        base_mask = corpus.expected_exported_base_mask(raw_mask)
        labels = input_ids[2:] + [0xFFFF_FFFF, 0xFFFF_FFFF]
        self.assertEqual(base_mask, [1, 1, 0, 0])
        self.assertEqual(labels, [201, 202, 0xFFFF_FFFF, 0xFFFF_FFFF])
        self.assertEqual(
            [labels[index] for index, active in enumerate(base_mask) if active],
            [201, 202],
        )

    def test_build_is_byte_reproducible_and_payload_tampering_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "first"
            second = root / "second"
            first_manifest = corpus.build(
                catalog_path=CATALOG_PATH,
                policy_path=POLICY_PATH,
                schema_path=SCHEMA_PATH,
                output=first,
                profile="pilot",
                forbidden_path=None,
                expected_forbidden_sha256=None,
            )
            second_manifest = corpus.build(
                catalog_path=CATALOG_PATH,
                policy_path=POLICY_PATH,
                schema_path=SCHEMA_PATH,
                output=second,
                profile="pilot",
                forbidden_path=None,
                expected_forbidden_sha256=None,
            )
            self.assertEqual(first_manifest, second_manifest)
            for name in ("train.jobs.jsonl", "eval.jobs.jsonl", "manifest.json", "SHA256SUMS"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            corpus.audit_directory(
                corpus_dir=first,
                catalog_path=CATALOG_PATH,
                policy_path=POLICY_PATH,
                forbidden_path=None,
                expected_forbidden_sha256=None,
            )
            with (first / "train.jobs.jsonl").open("ab") as stream:
                stream.write(b" ")
            with self.assertRaisesRegex(corpus.CorpusError, "payload hash mismatch"):
                corpus.audit_directory(
                    corpus_dir=first,
                    catalog_path=CATALOG_PATH,
                    policy_path=POLICY_PATH,
                    forbidden_path=None,
                    expected_forbidden_sha256=None,
                )

    def test_manifest_exposes_current_exporter_bridge_without_claiming_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "corpus"
            manifest = corpus.build(
                catalog_path=CATALOG_PATH,
                policy_path=POLICY_PATH,
                schema_path=SCHEMA_PATH,
                output=output,
                profile="pilot",
                forbidden_path=None,
                expected_forbidden_sha256=None,
            )
            contract = manifest["materialization_contract"]
            self.assertEqual(contract["status"], "required_not_performed")
            self.assertEqual(contract["exporter_input"], "one JSONL object per job: {id,input_ids,loss_mask}")
            self.assertIn("input_ids[Q+1]", contract["raw_loss_mask"])
            self.assertIn("raw_loss_mask[P+1]", contract["exported_loss_mask"])
            self.assertIn("token P+2", contract["exported_loss_mask"])
            self.assertEqual(manifest["forbidden_reference"]["status"], "not_supplied_policy_only")
            self.assertEqual(
                manifest["forbidden_reference"]["expected_sha256"],
                "7c7319ab066e8e6f5bb83a813082db1ee117aacd88405a36616b7e54dd0de6a2",
            )
            self.assertIsNone(manifest["forbidden_reference"]["audited_sha256"])
            sealed = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(sealed, manifest)


if __name__ == "__main__":
    unittest.main()
