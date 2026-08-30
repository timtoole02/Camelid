#!/usr/bin/env python3
"""Build and audit a deterministic pre-materialization EAGLE-3 corpus.

The output is intentionally a job stream, not tokenized training data.  A target-authoritative
materializer must render each chat, greedily generate the assistant completion with the pinned
Q4 target, and turn the combined token stream into the current Rust exporter's
``{id,input_ids,loss_mask}`` shape.  See README.md beside this file.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import string
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence


JOB_SCHEMA = "camelid-eagle3-corpus-job-v1"
MANIFEST_SCHEMA = "camelid-eagle3-corpus-manifest-v1"
SPLITS = ("train", "eval")
CATEGORIES = ("technical_instructional", "code_system_design", "general")
WORD_RE = re.compile(r"[a-z0-9]+")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "also",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "by",
        "can",
        "do",
        "for",
        "from",
        "give",
        "have",
        "how",
        "i",
        "if",
        "in",
        "include",
        "into",
        "is",
        "it",
        "let",
        "make",
        "of",
        "on",
        "or",
        "our",
        "should",
        "show",
        "that",
        "the",
        "their",
        "this",
        "to",
        "use",
        "user",
        "we",
        "what",
        "when",
        "where",
        "which",
        "with",
        "write",
        "you",
    }
)


class CorpusError(ValueError):
    """A deterministic corpus contract failed."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CorpusError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise CorpusError(f"expected a JSON object in {path}")
    return value


def normalized_words(text: str) -> list[str]:
    return WORD_RE.findall(text.casefold())


def normalized_text(text: str) -> str:
    return " ".join(normalized_words(text))


def content_words(text: str) -> list[str]:
    return [word for word in normalized_words(text) if word not in STOPWORDS]


def content_digest(messages: Sequence[Mapping[str, str]]) -> str:
    payload = [{"role": item["role"], "content": item["content"]} for item in messages]
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def raw_assistant_loss_mask(total_tokens: int, assistant_start: int) -> list[int]:
    """Build the causal next-token mask expected by the raw exporter input.

    Raw row ``Q`` predicts ``input_ids[Q+1]``.  The final row has no next token and is always
    zero.  The exporter, not this function, later shifts this raw mask into its P+2 EAGLE base
    rows.
    """

    if total_tokens < 2:
        raise CorpusError("raw assistant mask needs at least two input tokens")
    if assistant_start <= 0 or assistant_start >= total_tokens:
        raise CorpusError(
            f"assistant_start must be in 1..{total_tokens - 1}, got {assistant_start}"
        )
    return [
        int(row + 1 >= assistant_start) if row + 1 < total_tokens else 0
        for row in range(total_tokens)
    ]


def expected_exported_base_mask(raw_loss_mask: Sequence[int]) -> list[int]:
    """Mirror the pinned exporter boundary for an integration-contract test.

    This is not a second implementation of feature export.  It makes the corpus-side expectation
    executable: base row ``P`` predicts token ``P+2`` and therefore consumes raw mask ``P+1``.
    """

    if len(raw_loss_mask) < 2 or any(value not in (0, 1) for value in raw_loss_mask):
        raise CorpusError("raw loss mask must contain at least two boolean rows")
    return list(raw_loss_mask[1:]) + [0]


def placeholders(template: str) -> tuple[str, ...]:
    names: list[str] = []
    for _, field_name, _, _ in string.Formatter().parse(template):
        if field_name is not None and field_name not in names:
            names.append(field_name)
    unknown = set(names) - {"subject", "context", "audience", "style"}
    if unknown:
        raise CorpusError(f"template has unsupported placeholders: {sorted(unknown)}")
    return tuple(names)


def validate_catalog(catalog: Mapping[str, Any]) -> None:
    if catalog.get("schema") != "camelid-eagle3-corpus-catalog-v1":
        raise CorpusError("unsupported corpus catalog schema")
    families = catalog.get("families")
    if not isinstance(families, list) or not families:
        raise CorpusError("catalog must contain families")
    seen_ids: set[str] = set()
    seen_templates: set[str] = set()
    for family in families:
        if not isinstance(family, dict):
            raise CorpusError("each family must be an object")
        family_id = family.get("id")
        template_id = family.get("template_id")
        if not isinstance(family_id, str) or family_id in seen_ids:
            raise CorpusError(f"duplicate or missing family id: {family_id!r}")
        if not isinstance(template_id, str) or template_id in seen_templates:
            raise CorpusError(f"duplicate or missing template id: {template_id!r}")
        seen_ids.add(family_id)
        seen_templates.add(template_id)
        if family.get("split") not in SPLITS:
            raise CorpusError(f"family {family_id} has invalid split")
        if family.get("category") not in CATEGORIES:
            raise CorpusError(f"family {family_id} has invalid category")
        if not isinstance(family.get("count"), int) or family["count"] <= 0:
            raise CorpusError(f"family {family_id} has invalid count")
        if not isinstance(family.get("subjects"), list) or not family["subjects"]:
            raise CorpusError(f"family {family_id} has no subjects")
        placeholders(str(family.get("template", "")))


def family_axes(family: Mapping[str, Any], shared: Mapping[str, Any]) -> dict[str, list[str]]:
    split = family["split"]
    axes: dict[str, list[str]] = {"subject": list(family["subjects"])}
    if "context" in placeholders(family["template"]):
        axes["context"] = list(
            family.get("extra_contexts", shared[f"{split}_contexts"])
        )
    if "audience" in placeholders(family["template"]):
        axes["audience"] = list(shared[f"{split}_audiences"])
    if "style" in placeholders(family["template"]):
        axes["style"] = list(shared[f"{split}_styles"])
    return axes


def expand_family(family: Mapping[str, Any], catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    axes = family_axes(family, catalog["shared_axes"])
    axis_names = tuple(axes)
    combinations = list(itertools.product(*(axes[name] for name in axis_names)))
    expected = int(family["count"])
    if len(combinations) != expected:
        raise CorpusError(
            f"family {family['id']} expands to {len(combinations)} records, expected {expected}"
        )
    license_info = catalog["license"]
    records: list[dict[str, Any]] = []
    for ordinal, values in enumerate(combinations):
        substitutions = dict(zip(axis_names, values, strict=True))
        user_text = family["template"].format(**substitutions)
        messages = [
            {"role": "system", "content": family["system"]},
            {"role": "user", "content": user_text},
        ]
        digest = content_digest(messages)
        source_record_key = ".".join(
            f"{name[0]}{axes[name].index(value):02d}"
            for name, value in zip(axis_names, values, strict=True)
        )
        category_short = {
            "technical_instructional": "tech",
            "code_system_design": "code",
            "general": "general",
        }[family["category"]]
        records.append(
            {
                "schema": JOB_SCHEMA,
                "id": (
                    f"{family['split']}.{category_short}.{family['id']}."
                    f"{ordinal:04d}.{digest[:12]}"
                ),
                "split": family["split"],
                "category": family["category"],
                "source": {
                    "source_id": f"camelid.local.{family['split']}.{family['id']}.v1",
                    "source_record_key": source_record_key,
                    "family_id": family["id"],
                    "template_id": family["template_id"],
                    "provenance": "deterministic-local-generation",
                    "license_spdx": license_info["spdx"],
                    "license_name": license_info["name"],
                    "notice": license_info["notice"],
                },
                "messages": messages,
                "generation": {
                    "method": "target_greedy",
                    "temperature": 0.0,
                    "max_new_tokens": family["max_new_tokens"],
                },
                "supervision": {
                    "scope": "assistant_completion_only",
                    "materialization": "exact_q4_target_required",
                },
                "content_sha256": digest,
            }
        )
    return records


def all_standard_records(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for family in sorted(catalog["families"], key=lambda item: item["id"]):
        records.extend(expand_family(family, catalog))
    return sorted(records, key=record_sort_key)


def record_sort_key(record: Mapping[str, Any]) -> tuple[int, int, str, str]:
    return (
        SPLITS.index(record["split"]),
        CATEGORIES.index(record["category"]),
        record["source"]["family_id"],
        record["id"],
    )


def select_profile(
    records: Sequence[dict[str, Any]], catalog: Mapping[str, Any], profile: str
) -> list[dict[str, Any]]:
    profiles = catalog["profiles"]
    if profile not in profiles:
        raise CorpusError(f"unknown profile {profile!r}; choose from {sorted(profiles)}")
    if profile == "standard":
        selected = list(records)
    else:
        selected = []
        expected = profiles[profile]["expected"]
        for split in SPLITS:
            for category in CATEGORIES:
                quota = int(expected[split][category])
                by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for record in records:
                    if record["split"] == split and record["category"] == category:
                        by_family[record["source"]["family_id"]].append(record)
                family_ids = sorted(by_family)
                if not family_ids:
                    raise CorpusError(f"no families for {split}/{category}")
                for family_records in by_family.values():
                    family_records.sort(key=record_sort_key)
                cursors = {family_id: 0 for family_id in family_ids}
                while quota > 0:
                    progressed = False
                    for family_id in family_ids:
                        cursor = cursors[family_id]
                        family_records = by_family[family_id]
                        if cursor >= len(family_records):
                            continue
                        selected.append(family_records[cursor])
                        cursors[family_id] += 1
                        quota -= 1
                        progressed = True
                        if quota == 0:
                            break
                    if not progressed:
                        raise CorpusError(f"profile quota exceeds records for {split}/{category}")
    return sorted(selected, key=record_sort_key)


def validate_job_shape(record: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "id",
        "split",
        "category",
        "source",
        "messages",
        "generation",
        "supervision",
        "content_sha256",
    }
    if set(record) != required:
        raise CorpusError(f"record {record.get('id')!r} fields differ from schema")
    if record["schema"] != JOB_SCHEMA:
        raise CorpusError(f"record {record.get('id')!r} has wrong schema")
    if record["split"] not in SPLITS or record["category"] not in CATEGORIES:
        raise CorpusError(f"record {record.get('id')!r} has invalid split/category")
    messages = record["messages"]
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or [item.get("role") for item in messages] != ["system", "user"]
        or any(set(item) != {"role", "content"} for item in messages)
        or any(not isinstance(item["content"], str) or not item["content"] for item in messages)
    ):
        raise CorpusError(f"record {record.get('id')!r} has invalid messages")
    source = record["source"]
    expected_source_keys = {
        "source_id",
        "source_record_key",
        "family_id",
        "template_id",
        "provenance",
        "license_spdx",
        "license_name",
        "notice",
    }
    if set(source) != expected_source_keys:
        raise CorpusError(f"record {record.get('id')!r} has invalid source fields")
    if source["provenance"] != "deterministic-local-generation":
        raise CorpusError(f"record {record.get('id')!r} has unapproved provenance")
    if source["license_spdx"] != "MIT":
        raise CorpusError(f"record {record.get('id')!r} has unapproved license")
    generation = record["generation"]
    if set(generation) != {"method", "temperature", "max_new_tokens"}:
        raise CorpusError(f"record {record.get('id')!r} has invalid generation fields")
    if generation["method"] != "target_greedy" or generation["temperature"] != 0.0:
        raise CorpusError(f"record {record.get('id')!r} is not exact greedy")
    if record["supervision"] != {
        "scope": "assistant_completion_only",
        "materialization": "exact_q4_target_required",
    }:
        raise CorpusError(f"record {record.get('id')!r} has invalid supervision")
    digest = content_digest(messages)
    if record["content_sha256"] != digest:
        raise CorpusError(f"record {record.get('id')!r} content hash mismatch")


def ngrams(words: Sequence[str], width: int) -> set[tuple[str, ...]]:
    if len(words) < width:
        return set()
    return {tuple(words[index : index + width]) for index in range(len(words) - width + 1)}


def maximum_window_overlap(candidate: Sequence[str], protected: Sequence[str]) -> float:
    """Return the largest content-word overlap against a similarly sized protected window."""

    candidate_set = set(candidate)
    if not candidate_set or not protected:
        return 0.0
    width = max(len(candidate), 1)
    if len(protected) <= width:
        return len(candidate_set & set(protected)) / len(candidate_set)
    best = 0.0
    stride = max(1, width // 4)
    starts = list(range(0, len(protected) - width + 1, stride))
    last = len(protected) - width
    if not starts or starts[-1] != last:
        starts.append(last)
    for start in starts:
        overlap = len(candidate_set & set(protected[start : start + width])) / len(candidate_set)
        best = max(best, overlap)
    return best


def audit_records(
    records: Sequence[Mapping[str, Any]],
    *,
    catalog: Mapping[str, Any],
    policy: Mapping[str, Any],
    profile: str,
    forbidden_text: str | None = None,
) -> dict[str, Any]:
    validate_catalog(catalog)
    family_map = {family["id"]: family for family in catalog["families"]}
    ids: set[str] = set()
    content_hashes: set[str] = set()
    normalized_prompts: set[str] = set()
    source_keys: set[tuple[str, str]] = set()
    split_sources: dict[str, set[str]] = defaultdict(set)
    split_families: dict[str, set[str]] = defaultdict(set)
    split_templates: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    compiled_rules = [re.compile(pattern, re.IGNORECASE) for pattern in policy["deny_regex"]]
    protected_words = normalized_words(forbidden_text or "")
    protected_content = content_words(forbidden_text or "")
    protected_ngrams = ngrams(protected_words, int(policy["forbidden_ngram_words"]))
    max_user_chars = 0
    min_user_chars = sys.maxsize
    max_user_words = 0
    min_user_words = sys.maxsize

    for record in records:
        validate_job_shape(record)
        record_id = record["id"]
        if record_id in ids:
            raise CorpusError(f"duplicate record id {record_id}")
        ids.add(record_id)
        digest = record["content_sha256"]
        if digest in content_hashes:
            raise CorpusError(f"duplicate message content at {record_id}")
        content_hashes.add(digest)
        user_text = record["messages"][1]["content"]
        prompt_norm = normalized_text(user_text)
        if prompt_norm in normalized_prompts:
            raise CorpusError(f"duplicate normalized user prompt at {record_id}")
        normalized_prompts.add(prompt_norm)
        user_chars = len(user_text)
        user_words = len(normalized_words(user_text))
        min_user_chars = min(min_user_chars, user_chars)
        max_user_chars = max(max_user_chars, user_chars)
        min_user_words = min(min_user_words, user_words)
        max_user_words = max(max_user_words, user_words)
        if not policy["min_user_chars"] <= user_chars <= policy["max_user_chars"]:
            raise CorpusError(f"record {record_id} has {user_chars} user chars outside policy")
        if not policy["min_user_words"] <= user_words <= policy["max_user_words"]:
            raise CorpusError(f"record {record_id} has {user_words} user words outside policy")
        max_new_tokens = record["generation"]["max_new_tokens"]
        if not policy["min_max_new_tokens"] <= max_new_tokens <= policy["max_max_new_tokens"]:
            raise CorpusError(f"record {record_id} has invalid max_new_tokens {max_new_tokens}")
        combined = "\n".join(item["content"] for item in record["messages"])
        for rule in compiled_rules:
            if rule.search(combined):
                raise CorpusError(f"record {record_id} matches leakage rule {rule.pattern!r}")
        if protected_words:
            overlap = ngrams(normalized_words(combined), int(policy["forbidden_ngram_words"]))
            shared = overlap & protected_ngrams
            if shared:
                phrase = " ".join(sorted(shared)[0])
                raise CorpusError(
                    f"record {record_id} shares a protected exact n-gram: {phrase!r}"
                )
            candidate_content = content_words(combined)
            if len(candidate_content) >= policy["near_window_min_content_words"]:
                score = maximum_window_overlap(candidate_content, protected_content)
                if score >= policy["near_window_overlap_threshold"]:
                    raise CorpusError(
                        f"record {record_id} protected-content overlap {score:.3f} exceeds policy"
                    )
        source = record["source"]
        family_id = source["family_id"]
        family = family_map.get(family_id)
        if family is None:
            raise CorpusError(f"record {record_id} names unknown family {family_id}")
        if family["split"] != record["split"] or family["category"] != record["category"]:
            raise CorpusError(f"record {record_id} differs from its preassigned family split")
        if family["template_id"] != source["template_id"]:
            raise CorpusError(f"record {record_id} differs from its family template")
        expected_source_id = f"camelid.local.{record['split']}.{family_id}.v1"
        if source["source_id"] != expected_source_id:
            raise CorpusError(
                f"record {record_id} source id is {source['source_id']!r}, "
                f"expected {expected_source_id!r}"
            )
        if (
            source["license_spdx"] != catalog["license"]["spdx"]
            or source["license_name"] != catalog["license"]["name"]
            or source["notice"] != catalog["license"]["notice"]
        ):
            raise CorpusError(f"record {record_id} license metadata differs from the catalog")
        source_key = (source["source_id"], source["source_record_key"])
        if source_key in source_keys:
            raise CorpusError(f"duplicate source record key at {record_id}")
        source_keys.add(source_key)
        split = record["split"]
        split_sources[split].add(source["source_id"])
        split_families[split].add(family_id)
        split_templates[split].add(source["template_id"])
        counts[split][record["category"]] += 1

    for label, split_sets in (
        ("source", split_sources),
        ("family", split_families),
        ("template", split_templates),
    ):
        overlap = split_sets["train"] & split_sets["eval"]
        if overlap:
            raise CorpusError(f"train/eval {label} overlap: {sorted(overlap)}")
    expected = catalog["profiles"][profile]["expected"]
    actual = {
        split: {category: counts[split][category] for category in CATEGORIES}
        for split in SPLITS
    }
    if actual != expected:
        raise CorpusError(f"profile counts differ: expected {expected}, got {actual}")
    total = len(records)
    category_totals = {
        category: sum(counts[split][category] for split in SPLITS) for category in CATEGORIES
    }
    return {
        "status": "pass",
        "record_count": total,
        "counts": actual,
        "category_ratios": {
            category: round(category_totals[category] / total, 6) for category in CATEGORIES
        },
        "unique_record_ids": len(ids),
        "unique_content_hashes": len(content_hashes),
        "unique_normalized_prompts": len(normalized_prompts),
        "split_overlap": {"sources": 0, "families": 0, "templates": 0},
        "leakage_matches": 0,
        "lengths": {
            "user_chars_min": 0 if not records else min_user_chars,
            "user_chars_max": max_user_chars,
            "user_words_min": 0 if not records else min_user_words,
            "user_words_max": max_user_words,
        },
    }


def records_digest(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["content_sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_record(path: Path, count: int) -> dict[str, Any]:
    return {
        "file": path.name,
        "records": count,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(canonical_json(record))
            stream.write("\n")


def write_text_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def ensure_empty_output(output: Path) -> None:
    if output.exists():
        if not output.is_dir():
            raise CorpusError(f"output exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise CorpusError(f"output directory is not empty: {output}")
    else:
        output.mkdir(parents=True)


def build(
    *,
    catalog_path: Path,
    policy_path: Path,
    schema_path: Path,
    output: Path,
    profile: str,
    forbidden_path: Path | None,
    expected_forbidden_sha256: str | None,
) -> dict[str, Any]:
    catalog = load_json(catalog_path)
    policy = load_json(policy_path)
    validate_catalog(catalog)
    standard = all_standard_records(catalog)
    records = select_profile(standard, catalog, profile)
    forbidden_text = None
    forbidden_sha256 = None
    pinned_forbidden_sha256 = policy.get("protected_reference_sha256")
    if not isinstance(pinned_forbidden_sha256, str) or not SHA256_RE.fullmatch(
        pinned_forbidden_sha256
    ):
        raise CorpusError("leakage policy must pin a lowercase protected-reference SHA-256")
    if (
        expected_forbidden_sha256 is not None
        and expected_forbidden_sha256 != pinned_forbidden_sha256
    ):
        raise CorpusError(
            "--expected-forbidden-sha256 differs from the versioned leakage policy; "
            "update the policy instead of overriding the canary"
        )
    if forbidden_path is not None:
        forbidden_sha256 = sha256_file(forbidden_path)
        if forbidden_sha256 != pinned_forbidden_sha256:
            raise CorpusError(
                f"forbidden reference SHA-256 mismatch: expected {pinned_forbidden_sha256}, "
                f"got {forbidden_sha256}"
            )
        forbidden_text = forbidden_path.read_text(encoding="utf-8")
    elif expected_forbidden_sha256 is not None:
        raise CorpusError("--expected-forbidden-sha256 requires --forbidden-file")
    audit = audit_records(
        records,
        catalog=catalog,
        policy=policy,
        profile=profile,
        forbidden_text=forbidden_text,
    )
    ensure_empty_output(output)
    by_split = {split: [record for record in records if record["split"] == split] for split in SPLITS}
    split_paths = {split: output / f"{split}.jobs.jsonl" for split in SPLITS}
    for split in SPLITS:
        write_jsonl(split_paths[split], by_split[split])
    family_counts = Counter(record["source"]["family_id"] for record in records)
    family_assignments = []
    family_map = {family["id"]: family for family in catalog["families"]}
    for family_id in sorted(family_counts):
        family = family_map[family_id]
        family_assignments.append(
            {
                "family_id": family_id,
                "source_id": f"camelid.local.{family['split']}.{family_id}.v1",
                "template_id": family["template_id"],
                "split": family["split"],
                "category": family["category"],
                "records": family_counts[family_id],
                "license_spdx": catalog["license"]["spdx"],
            }
        )
    tool_dir = Path(__file__).resolve().parent
    generator_inputs = {}
    for name, path in (
        ("builder", Path(__file__).resolve()),
        ("catalog", catalog_path.resolve()),
        ("leakage_policy", policy_path.resolve()),
        ("job_schema", schema_path.resolve()),
    ):
        generator_inputs[name] = {"file": path.name, "sha256": sha256_file(path)}
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "corpus_job_schema": JOB_SCHEMA,
        "profile": profile,
        "generator_version": catalog["generator_version"],
        "determinism": {
            "ordering": "split,category,family,id; canonical sorted-key JSONL",
            "randomness": "none",
            "generator_inputs": generator_inputs,
            "records_digest": records_digest(records),
        },
        "split_strategy": {
            "method": "source/template/family assigned to one split in catalog before expansion",
            "expansion": "ordered Cartesian product of declared axes",
            "train_eval_source_overlap": 0,
            "train_eval_template_overlap": 0,
            "train_eval_family_overlap": 0,
        },
        "materialization_contract": {
            "status": "required_not_performed",
            "target": "pinned exact-Q4 Llama-3.2-3B-Instruct greedy decode",
            "temperature": 0.0,
            "chat_rendering": "target-native template; record exact rendered bytes and token ids",
            "exporter_input": "one JSONL object per job: {id,input_ids,loss_mask}",
            "raw_loss_mask": "raw_loss_mask[Q]=1 exactly when input_ids[Q+1] is a target-generated assistant token; otherwise 0",
            "exported_loss_mask": "exporter shifts once: base_loss_mask[P]=raw_loss_mask[P+1], then zeros the final row because base row P predicts token P+2",
            "capture_replay": "a teacher-forced replay is permitted only to capture exact-Q4 features when capture was not fused into generation",
            "prohibition": "no reference answer, cached response, prepared continuation, alternate teacher, or protected-canary material",
        },
        "license": catalog["license"],
        "forbidden_reference": {
            "status": "audited" if forbidden_sha256 else "not_supplied_policy_only",
            "expected_sha256": pinned_forbidden_sha256,
            "audited_sha256": forbidden_sha256,
            "path_recorded": False,
        },
        "files": {split: file_record(split_paths[split], len(by_split[split])) for split in SPLITS},
        "family_assignments": family_assignments,
        "audit": audit,
    }
    manifest_path = output / "manifest.json"
    write_text_exclusive(manifest_path, json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    checksum_lines = [
        f"{sha256_file(split_paths[split])}  {split_paths[split].name}" for split in SPLITS
    ]
    checksum_lines.append(f"{sha256_file(manifest_path)}  manifest.json")
    write_text_exclusive(output / "SHA256SUMS", "\n".join(checksum_lines) + "\n")
    return manifest


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CorpusError(f"{path}:{line_number} is not a JSON object")
                records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise CorpusError(f"cannot read JSONL {path}: {error}") from error
    return records


def audit_directory(
    *,
    corpus_dir: Path,
    catalog_path: Path,
    policy_path: Path,
    forbidden_path: Path | None,
    expected_forbidden_sha256: str | None,
) -> dict[str, Any]:
    manifest = load_json(corpus_dir / "manifest.json")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise CorpusError("unsupported corpus manifest schema")
    catalog = load_json(catalog_path)
    policy = load_json(policy_path)
    records: list[dict[str, Any]] = []
    for split in SPLITS:
        metadata = manifest["files"][split]
        path = corpus_dir / metadata["file"]
        if sha256_file(path) != metadata["sha256"]:
            raise CorpusError(f"payload hash mismatch for {path.name}")
        split_records = read_jsonl(path)
        if len(split_records) != metadata["records"]:
            raise CorpusError(f"record count mismatch for {path.name}")
        if any(record.get("split") != split for record in split_records):
            raise CorpusError(f"wrong-split record in {path.name}")
        records.extend(split_records)
    forbidden_text = None
    forbidden_sha256 = None
    pinned_forbidden_sha256 = policy.get("protected_reference_sha256")
    if not isinstance(pinned_forbidden_sha256, str) or not SHA256_RE.fullmatch(
        pinned_forbidden_sha256
    ):
        raise CorpusError("leakage policy must pin a lowercase protected-reference SHA-256")
    if (
        expected_forbidden_sha256 is not None
        and expected_forbidden_sha256 != pinned_forbidden_sha256
    ):
        raise CorpusError(
            "--expected-forbidden-sha256 differs from the versioned leakage policy"
        )
    if forbidden_path is not None:
        forbidden_sha256 = sha256_file(forbidden_path)
        if forbidden_sha256 != pinned_forbidden_sha256:
            raise CorpusError("forbidden reference SHA-256 mismatch")
        forbidden_text = forbidden_path.read_text(encoding="utf-8")
    elif expected_forbidden_sha256 is not None:
        raise CorpusError("--expected-forbidden-sha256 requires --forbidden-file")
    result = audit_records(
        records,
        catalog=catalog,
        policy=policy,
        profile=manifest["profile"],
        forbidden_text=forbidden_text,
    )
    if result != manifest["audit"]:
        raise CorpusError("fresh audit differs from sealed manifest audit")
    sealed_forbidden = manifest["forbidden_reference"]
    if sealed_forbidden.get("expected_sha256") != pinned_forbidden_sha256:
        raise CorpusError("manifest protected-reference pin differs from leakage policy")
    if sealed_forbidden.get("status") == "audited":
        if forbidden_sha256 is None:
            raise CorpusError("sealed corpus requires the protected file for re-audit")
        if sealed_forbidden.get("audited_sha256") != forbidden_sha256:
            raise CorpusError("protected reference differs from sealed corpus audit")
    if records_digest(sorted(records, key=record_sort_key)) != manifest["determinism"]["records_digest"]:
        raise CorpusError("ordered record digest differs from manifest")
    checksum_path = corpus_dir / "SHA256SUMS"
    expected_checksums = {
        metadata["file"]: metadata["sha256"] for metadata in manifest["files"].values()
    }
    expected_checksums["manifest.json"] = sha256_file(corpus_dir / "manifest.json")
    try:
        checksum_lines = checksum_path.read_text(encoding="utf-8").splitlines()
        observed_checksums = {}
        for line in checksum_lines:
            digest, filename = line.split("  ", 1)
            if filename in observed_checksums:
                raise CorpusError(f"duplicate SHA256SUMS entry for {filename}")
            observed_checksums[filename] = digest
    except (OSError, ValueError) as error:
        raise CorpusError(f"cannot parse {checksum_path}: {error}") from error
    if observed_checksums != expected_checksums:
        raise CorpusError("SHA256SUMS differs from the sealed corpus files")
    return {
        "status": "pass",
        "profile": manifest["profile"],
        "records": len(records),
        "forbidden_reference_sha256": forbidden_sha256,
    }


def parser(default_dir: Path) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--catalog", type=Path, default=default_dir / "catalog.json", help=argparse.SUPPRESS
    )
    result.add_argument(
        "--policy",
        type=Path,
        default=default_dir / "leakage_rules.json",
        help=argparse.SUPPRESS,
    )
    subparsers = result.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="build a new deterministic corpus directory")
    build_parser.add_argument("--profile", choices=("pilot", "standard"), default="pilot")
    build_parser.add_argument("--output", required=True, type=Path)
    build_parser.add_argument("--forbidden-file", type=Path)
    build_parser.add_argument("--expected-forbidden-sha256")
    audit_parser = subparsers.add_parser("audit", help="re-audit an existing corpus directory")
    audit_parser.add_argument("--corpus-dir", required=True, type=Path)
    audit_parser.add_argument("--forbidden-file", type=Path)
    audit_parser.add_argument("--expected-forbidden-sha256")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    tool_dir = Path(__file__).resolve().parent
    args = parser(tool_dir).parse_args(argv)
    try:
        if args.expected_forbidden_sha256 is not None and not SHA256_RE.fullmatch(
            args.expected_forbidden_sha256
        ):
            raise CorpusError("--expected-forbidden-sha256 must be 64 lowercase hex characters")
        if args.command == "build":
            manifest = build(
                catalog_path=args.catalog,
                policy_path=args.policy,
                schema_path=tool_dir / "corpus_job.schema.json",
                output=args.output,
                profile=args.profile,
                forbidden_path=args.forbidden_file,
                expected_forbidden_sha256=args.expected_forbidden_sha256,
            )
            print(
                canonical_json(
                    {
                        "status": "pass",
                        "profile": args.profile,
                        "output": str(args.output),
                        "records": manifest["audit"]["record_count"],
                        "records_digest": manifest["determinism"]["records_digest"],
                    }
                )
            )
        else:
            result = audit_directory(
                corpus_dir=args.corpus_dir,
                catalog_path=args.catalog,
                policy_path=args.policy,
                forbidden_path=args.forbidden_file,
                expected_forbidden_sha256=args.expected_forbidden_sha256,
            )
            print(canonical_json(result))
    except CorpusError as error:
        print(f"corpus gate failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
