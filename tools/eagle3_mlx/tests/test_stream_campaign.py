from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "stream_campaign.py"
SPEC = importlib.util.spec_from_file_location("stream_campaign_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
stream_campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stream_campaign)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


class StreamCampaignTests(unittest.TestCase):
    def test_discovers_and_hash_verifies_materialized_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = []
            total = 0
            for index, records in enumerate((3, 2)):
                shard = root / f"shard-{index:06d}"
                shard.mkdir()
                records_path = shard / "records.jsonl"
                records_path.write_text("".join(f'{{"id":"{index}-{row}"}}\n' for row in range(records)), encoding="utf-8")
                manifest = {
                    "schema": stream_campaign.MATERIALIZATION_SHARD_SCHEMA,
                    "index": index,
                    "records": {
                        "file": "records.jsonl",
                        "records": records,
                        "sha256": stream_campaign.sha256_file(records_path),
                    },
                }
                _write_json(shard / "manifest.json", manifest)
                manifests.append(stream_campaign.sha256_file(shard / "manifest.json"))
                total += records
            run = {
                "schema": stream_campaign.MATERIALIZATION_RUN_SCHEMA,
                "source": {"records": total, "sha256": "a" * 64},
                "target": {"gguf_sha256": stream_campaign.TARGET_SHA256},
                "runtime": {"binary_sha256": "b" * 64, "camelid_commit": "c" * 40},
            }
            _write_json(root / "run.json", run)
            complete = {
                "schema": stream_campaign.MATERIALIZATION_COMPLETE_SCHEMA,
                "run_manifest_sha256": stream_campaign.sha256_file(root / "run.json"),
                "records": total,
                "shards": 2,
                "shard_manifest_sha256": manifests,
            }
            _write_json(root / "COMPLETE.json", complete)
            shards, seal = stream_campaign.discover_materialized_shards(root)
            self.assertEqual([shard["records"] for shard in shards], [3, 2])
            self.assertEqual(seal["records"], 5)
            self.assertEqual(seal["source_jobs_sha256"], "a" * 64)

    def test_materialized_payload_tampering_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "shard-000000"
            shard.mkdir()
            records = shard / "records.jsonl"
            records.write_text('{"id":"a"}\n', encoding="utf-8")
            manifest = {
                "schema": stream_campaign.MATERIALIZATION_SHARD_SCHEMA,
                "index": 0,
                "records": {"file": "records.jsonl", "records": 1, "sha256": "0" * 64},
            }
            _write_json(shard / "manifest.json", manifest)
            run = {
                "schema": stream_campaign.MATERIALIZATION_RUN_SCHEMA,
                "source": {"records": 1, "sha256": "a" * 64},
                "target": {"gguf_sha256": stream_campaign.TARGET_SHA256},
                "runtime": {"binary_sha256": "b" * 64, "camelid_commit": "c" * 40},
            }
            _write_json(root / "run.json", run)
            _write_json(
                root / "COMPLETE.json",
                {
                    "schema": stream_campaign.MATERIALIZATION_COMPLETE_SCHEMA,
                    "run_manifest_sha256": stream_campaign.sha256_file(root / "run.json"),
                    "records": 1,
                    "shards": 1,
                    "shard_manifest_sha256": [stream_campaign.sha256_file(shard / "manifest.json")],
                },
            )
            with self.assertRaises(stream_campaign.CampaignError):
                stream_campaign.discover_materialized_shards(root)

    def test_cleanup_is_confined_to_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inside = root / "features" / "shard-000000"
            inside.mkdir(parents=True)
            stream_campaign.safe_remove_regenerable(inside, root)
            self.assertFalse(inside.exists())
            outside = root.parent / "outside-campaign-test"
            with self.assertRaises(stream_campaign.CampaignError):
                stream_campaign.safe_remove_regenerable(outside, root)


if __name__ == "__main__":
    unittest.main()
