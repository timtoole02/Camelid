#!/usr/bin/env python3
"""Seal a deterministic, stratified subset of a built EAGLE-3 eval split."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .build_corpus import (
    CATEGORIES,
    CorpusError,
    JOB_SCHEMA,
    MANIFEST_SCHEMA,
    canonical_json,
    ensure_empty_output,
    file_record,
    load_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    write_jsonl,
    write_text_exclusive,
)


SUBSET_MANIFEST_SCHEMA = "camelid-eagle3-eval-subset-manifest-v1"


def _largest_remainder(total: int, populations: Mapping[str, int]) -> dict[str, int]:
    population_total = sum(populations.values())
    if total <= 0:
        raise CorpusError("subset count must be positive")
    if not populations or any(value <= 0 for value in populations.values()):
        raise CorpusError("allocation populations must be non-empty and positive")
    if total > population_total:
        raise CorpusError(
            f"cannot select {total} records from a population of {population_total}"
        )
    allocated = {
        key: total * population // population_total
        for key, population in populations.items()
    }
    remaining = total - sum(allocated.values())
    order = sorted(
        populations,
        key=lambda key: (-(total * populations[key] % population_total), key),
    )
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


def select_records(records: Sequence[dict[str, Any]], count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    category_counts = Counter(record["category"] for record in records)
    unknown_categories = set(category_counts) - set(CATEGORIES)
    if unknown_categories:
        raise CorpusError(f"unknown eval categories: {sorted(unknown_categories)}")
    category_quotas = _largest_remainder(count, category_counts)

    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        family_counts[record["category"]][record["source"]["family_id"]] += 1
    family_quotas = {
        category: _largest_remainder(category_quotas[category], family_counts[category])
        for category in category_quotas
    }

    selected: list[dict[str, Any]] = []
    used: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        category = record["category"]
        family = record["source"]["family_id"]
        if used[category][family] < family_quotas[category][family]:
            selected.append(record)
            used[category][family] += 1
    if len(selected) != count:
        raise CorpusError(f"selection produced {len(selected)} records, expected {count}")
    return selected, {
        "category_quotas": dict(sorted(category_quotas.items())),
        "family_quotas": {
            category: dict(sorted(quotas.items()))
            for category, quotas in sorted(family_quotas.items())
        },
    }


def build_subset(*, corpus_dir: Path, output: Path, count: int) -> dict[str, Any]:
    source_manifest_path = corpus_dir / "manifest.json"
    source_manifest = load_json(source_manifest_path)
    if source_manifest.get("schema") != MANIFEST_SCHEMA:
        raise CorpusError("unsupported source corpus manifest schema")
    eval_entry = source_manifest.get("files", {}).get("eval")
    if not isinstance(eval_entry, dict) or eval_entry.get("file") != "eval.jobs.jsonl":
        raise CorpusError("source manifest does not seal eval.jobs.jsonl")
    eval_path = corpus_dir / "eval.jobs.jsonl"
    actual_eval_sha = sha256_file(eval_path)
    if actual_eval_sha != eval_entry.get("sha256"):
        raise CorpusError(
            f"source eval SHA-256 is {actual_eval_sha}, expected {eval_entry.get('sha256')}"
        )
    records = read_jsonl(eval_path)
    if len(records) != eval_entry.get("records"):
        raise CorpusError("source eval record count differs from its manifest")
    ids = [record.get("id") for record in records]
    if len(ids) != len(set(ids)):
        raise CorpusError("source eval contains duplicate IDs")
    for record in records:
        if record.get("schema") != JOB_SCHEMA or record.get("split") != "eval":
            raise CorpusError("source eval contains a non-eval or unsupported job record")
        source = record.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("family_id"), str):
            raise CorpusError("source eval record is missing its family assignment")

    selected, quotas = select_records(records, count)
    ensure_empty_output(output)
    output_jobs = output / "eval.jobs.jsonl"
    write_jsonl(output_jobs, selected)
    selected_ids = [record["id"] for record in selected]
    manifest = {
        "schema": SUBSET_MANIFEST_SCHEMA,
        "source": {
            "corpus_manifest_sha256": sha256_file(source_manifest_path),
            "eval_jobs_sha256": actual_eval_sha,
            "eval_records": len(records),
            "profile": source_manifest.get("profile"),
            "forbidden_reference": source_manifest.get("forbidden_reference"),
        },
        "selection": {
            "count": count,
            "method": "category-proportional then family-proportional largest-remainder; first records per family; preserve source order",
            "randomness": "none",
            "source_order_preserved": True,
            "selected_ids_sha256": sha256_bytes(canonical_json(selected_ids).encode("utf-8")),
            **quotas,
        },
        "output": file_record(output_jobs, len(selected)),
    }
    manifest_path = output / "manifest.json"
    write_text_exclusive(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    write_text_exclusive(
        output / "SHA256SUMS",
        f"{sha256_file(output_jobs)}  eval.jobs.jsonl\n"
        f"{sha256_file(manifest_path)}  manifest.json\n",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        manifest = build_subset(
            corpus_dir=args.corpus_dir,
            output=args.output,
            count=args.count,
        )
        print(
            canonical_json(
                {
                    "status": "pass",
                    "output": str(args.output),
                    "records": manifest["selection"]["count"],
                    "jobs_sha256": manifest["output"]["sha256"],
                }
            )
        )
    except (CorpusError, OSError, KeyError, TypeError) as error:
        print(f"eval subset gate failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
