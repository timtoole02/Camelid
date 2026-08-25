#!/usr/bin/env python3
"""Fail closed and summarize one cap8/cap16/cap16/cap8 Mini2 campaign."""

import json
import os
import re
import sys
from decimal import Decimal, InvalidOperation

from summarize import read_env_manifest


SELECTOR = "CAMELID_GEMMA4_GHOST_METAL_LIVE_SEQUENTIAL_STAGE_CAP16"
EXPECTED_CAP16 = (False, True, True, False)
MTP_ROUND_RE = re.compile(
    r"\[mtp round\] #(\d+) wall=([\d.]+)ms "
    r"\(assistant=[\d.]+ms, verifier=[\d.]+ms\) accepted=(\d+)/(\d+)"
)
EXPECTED_MTP_WIDTHS = (14, 13, 14, 7)


class ReceiptError(ValueError):
    pass


def load_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ReceiptError(f"cannot read valid JSON from {path}: {error}") from error


def require(condition, message):
    if not condition:
        raise ReceiptError(message)


def decode_throughput(log_path):
    try:
        with open(log_path, encoding="utf-8", errors="strict") as handle:
            log = handle.read()
    except (OSError, UnicodeError) as error:
        raise ReceiptError(f"cannot read {log_path}: {error}") from error

    rounds = MTP_ROUND_RE.findall(log)
    require(len(rounds) == 4, f"expected exactly four MTP rounds in {log_path}")
    require(
        [int(index) for index, _, _, _ in rounds] == list(range(4)),
        f"MTP round indices are not exactly 0,1,2,3 in {log_path}",
    )
    require(
        tuple(int(width) for _, _, _, width in rounds) == EXPECTED_MTP_WIDTHS,
        f"MTP verifier widths are not {EXPECTED_MTP_WIDTHS} in {log_path}",
    )

    try:
        wall_ms = sum((Decimal(wall) for _, wall, _, _ in rounds), Decimal(0))
        accepted = sum(int(count) for _, _, count, _ in rounds)
    except (InvalidOperation, ValueError) as error:
        raise ReceiptError(f"malformed decode rounds in {log_path}") from error
    require(wall_ms > 0, f"nonpositive decode wall in {log_path}")

    count = len(rounds)
    emitted = accepted + count
    require(emitted == 48, f"MTP round accounting emitted {emitted}, not 48, in {log_path}")
    tok_s = Decimal(1000) * Decimal(emitted) / wall_ms
    return "mtp", count, tok_s


def validate_manual_safety(path):
    receipt = load_json(path)
    require(receipt.get("schema_version") == 1, f"bad manual safety schema in {path}")
    require(receipt.get("mode") == "manual-no-watchdog", f"wrong supervision in {path}")
    require(receipt.get("qualifying") is False, f"manual receipt qualified itself in {path}")
    require(receipt.get("point_samples_only") is True, f"manual receipt lost point-sample label in {path}")
    checks = receipt.get("checks")
    require(isinstance(checks, dict), f"missing manual checks in {path}")
    require(checks.get("all_passed") is True, f"manual checks failed in {path}")
    child = receipt.get("child")
    require(isinstance(child, dict), f"missing child result in {path}")
    require(child.get("returncode") == 0, f"child failed in {path}")
    require(child.get("process_group_empty") is True,
            f"child process group remained alive in {path}")


def validate_response(path):
    response = load_json(path)
    camelid = response.get("camelid")
    require(isinstance(camelid, dict), f"missing camelid response data in {path}")
    require(camelid.get("exact_match_expected") is True, f"token parity failed in {path}")
    require(camelid.get("exact_match_count") == 48, f"not exact 48/48 in {path}")
    usage = response.get("usage")
    require(isinstance(usage, dict), f"missing response usage in {path}")
    require(usage.get("completion_tokens") == 48,
            f"completion count is not 48 in {path}")
    generated = camelid.get("generated_token_ids")
    require(isinstance(generated, list) and len(generated) == 48,
            f"generated token vector is not length 48 in {path}")


def read_run(run_dir, expect_cap16):
    require(os.path.isdir(run_dir), f"not a run directory: {run_dir}")
    env_path = os.path.join(run_dir, "env.txt")
    try:
        env = read_env_manifest(env_path)
    except (OSError, UnicodeError, ValueError) as error:
        raise ReceiptError(f"invalid environment manifest {env_path}: {error}") from error

    require(env.get("manifest_format") == "base64-v1", f"legacy environment manifest in {env_path}")
    require(env.get("supervision_mode") == "manual-no-watchdog",
            f"run did not disable watchdog in {env_path}")
    require(env.get("source_tree_clean") == "1", f"source tree was dirty in {env_path}")
    require(env.get("CAMELID_GEMMA4_KV_INIT") == "192", f"KV_INIT drift in {env_path}")
    require(env.get("CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS_PER_LAYER") ==
            "60,64,45,43,40,41,44,42,47,46,40,41,44,46,42,50,40,39,46,47,41,46,45,43,46,56,59,56,58,51",
            f"1,408-slot profile drift in {env_path}")
    require(env.get("CAMELID_GEMMA4_GHOST_METAL_LIVE_SEQUENTIAL_STAGE_DUAL_READER") == "1",
            f"dual-reader lane drift in {env_path}")
    selector = env.get(SELECTOR)
    if expect_cap16:
        require(selector == "1", f"cap16 selector missing in {env_path}")
    else:
        require(selector is None, f"cap8 control contains cap16 selector in {env_path}")

    validate_response(os.path.join(run_dir, "response.json"))
    validate_manual_safety(os.path.join(run_dir, "manual-safety.json"))
    kind, rounds, tok_s = decode_throughput(os.path.join(run_dir, "server.log"))
    return {
        "run": os.path.basename(os.path.abspath(run_dir)),
        "profile": "cap16" if expect_cap16 else "cap8",
        "round_kind": kind,
        "rounds": rounds,
        "decode_tok_s": tok_s,
        "env": env,
    }


def printable_decimal(value):
    return float(value.quantize(Decimal("0.000001")))


def summarize_abba(run_dirs):
    require(len(run_dirs) == 4, "expected exactly four run directories")
    runs = [read_run(path, cap16) for path, cap16 in zip(run_dirs, EXPECTED_CAP16)]

    normalized = []
    for run in runs:
        manifest = dict(run["env"])
        manifest.pop(SELECTOR, None)
        normalized.append(manifest)
    require(all(manifest == normalized[0] for manifest in normalized[1:]),
            "ABBA environment or binary/source provenance drifted beyond the cap16 selector")

    control = (runs[0]["decode_tok_s"] + runs[3]["decode_tok_s"]) / Decimal(2)
    candidate = (runs[1]["decode_tok_s"] + runs[2]["decode_tok_s"]) / Decimal(2)
    ratio = candidate / control
    improvement_pct = (ratio - Decimal(1)) * Decimal(100)
    candidate_values = (runs[1]["decode_tok_s"], runs[2]["decode_tok_s"])

    return {
        "schema_version": 1,
        "valid": True,
        "supervision": "manual-no-watchdog",
        "qualifying_safety_evidence": False,
        "order": ["cap8", "cap16", "cap16", "cap8"],
        "source_commit": runs[0]["env"]["source_commit"],
        "binary_sha256": runs[0]["env"]["binary_sha256"],
        "runs": [
            {key: printable_decimal(value) if key == "decode_tok_s" else value
             for key, value in run.items() if key != "env"}
            for run in runs
        ],
        "cap8_mean_decode_tok_s": printable_decimal(control),
        "cap16_mean_decode_tok_s": printable_decimal(candidate),
        "cap16_ratio_vs_cap8": printable_decimal(ratio),
        "cap16_improvement_pct": printable_decimal(improvement_pct),
        "cap16_mean_at_least_50_tok_s": candidate >= Decimal(50),
        "both_cap16_runs_at_least_50_tok_s": all(value >= Decimal(50) for value in candidate_values),
    }


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 4:
        print(f"usage: {os.path.basename(sys.argv[0])} CAP8_A CAP16_B CAP16_B CAP8_A", file=sys.stderr)
        return 64
    try:
        result = summarize_abba(args)
    except ReceiptError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 75
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
