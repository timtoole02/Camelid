#!/usr/bin/env python3
"""Offline residency-identity coverage study.

Question: can a CALIBRATION-derived (other-prompts) 1,408-slot identity
profile cover enough of the frozen fixture's decode route occurrences to
close the gap to 35 tok/s, without fixture-oracle knowledge?

Linear model (same as analyze_exact_residency_trace.py): projected residual
wave = sum_r wave_ms_r * |route_r - profile| / cold_r_observed, where
cold_r_observed = |route_r - resident_at_start_r| from the trace.
"""
import re
import sys
from collections import defaultdict

LAYERS, EXPERTS = 30, 128
TRACE = re.compile(r"\[gemma4 exact-residency trace\] .*admitted=1 .*")


def field(line, name):
    m = re.search(rf"{name}=([^ ]+)", line)
    assert m, f"missing {name}"
    return m.group(1)


def masks(value):
    parts = value.split("/")
    assert len(parts) == LAYERS, f"expected {LAYERS} layers, got {len(parts)}"
    out = []
    for layer, part in enumerate(parts):
        m = re.fullmatch(rf"L{layer}:([0-9a-f]{{32}})", part)
        assert m, f"bad mask {part!r}"
        v = int(m.group(1), 16)
        out.append(frozenset(e for e in range(EXPERTS) if v & (1 << e)))
    return out


def parse_trace(path):
    rounds = []
    for line in open(path, errors="replace"):
        if "[gemma4 exact-residency trace]" not in line or "admitted=1" not in line:
            continue
        rounds.append({
            "K": int(field(line, "K")),
            "wave_ms": float(field(line, "total_wave_load_ms")),
            "cold": int(field(line, "cold_records")),
            "routes": masks(field(line, "route_masks")),
            "resident": masks(field(line, "resident_masks")),
        })
    return rounds


def coverage(fix_rounds, profile):
    """profile: list[frozenset] per layer. Returns (covered_occ, total_occ, residual_wave_ms)."""
    covered = total = 0
    residual_wave = 0.0
    for r in fix_rounds:
        residual_cold = 0
        for layer in range(LAYERS):
            for e in r["routes"][layer]:
                total += 1
                if e in profile[layer]:
                    covered += 1
                else:
                    residual_cold += 1
        if r["cold"] > 0:
            residual_wave += r["wave_ms"] * residual_cold / r["cold"]
    return covered, total, residual_wave


def build_profile(scores, capacities, prefer=None):
    """Top-capacity experts per layer by score desc, tie: prefer set, then low id."""
    prof = []
    for layer in range(LAYERS):
        cap = capacities[layer]
        keys = sorted(
            range(EXPERTS),
            key=lambda e: (
                -scores[layer].get(e, 0.0),
                0 if (prefer and e in prefer[layer]) else 1,
                e,
            ),
        )
        prof.append(frozenset(keys[:cap]))
    return prof


def main():
    fixture_path, *cal_paths = sys.argv[1:]
    fix = parse_trace(fixture_path)
    assert len(fix) == 4, f"fixture trace has {len(fix)} admitted rounds"
    capacities = [len(r) for r in fix[0]["resident"]]
    total_wave = sum(r["wave_ms"] for r in fix)

    # (a) current resident content (round-start residency, constant per trace)
    current = fix[0]["resident"]

    # (b) fixture-oracle: analyzer-style per-cold-record wave weighting
    oracle_scores = [defaultdict(float) for _ in range(LAYERS)]
    for r in fix:
        w = r["wave_ms"] / r["cold"] if r["cold"] else 0.0
        for layer in range(LAYERS):
            for e in r["routes"][layer]:
                oracle_scores[layer][e] += w
    oracle = build_profile(oracle_scores, capacities, prefer=current)

    # (c) calibration: occurrence counts across other prompts' decode rounds,
    # each prompt weighted equally (per-round occurrence / n_rounds).
    cal_scores = [defaultdict(float) for _ in range(LAYERS)]
    used_cals = []
    for p in cal_paths:
        rounds = parse_trace(p)
        if not rounds:
            print(f"WARNING: {p} produced no admitted trace rounds; skipped")
            continue
        used_cals.append((p, len(rounds)))
        for r in rounds:
            for layer in range(LAYERS):
                for e in r["routes"][layer]:
                    cal_scores[layer][e] += 1.0 / len(rounds)
    calib = build_profile(cal_scores, capacities)

    # (d) blend: calibration prior + current (prompt/LFU) content as tiebreak
    blend_scores = [defaultdict(float) for _ in range(LAYERS)]
    for layer in range(LAYERS):
        for e, s in cal_scores[layer].items():
            blend_scores[layer][e] += s
        for e in current[layer]:
            blend_scores[layer][e] += 0.5  # prompt/LFU membership as half-vote
    blend = build_profile(blend_scores, capacities)

    print(f"fixture rounds: {[r['K'] for r in fix]}, total wave {total_wave:.1f} ms, "
          f"capacity {sum(capacities)}")
    for name, prof in [("current", current), ("oracle", oracle),
                       ("calibration", calib), ("blend(cal+current)", blend)]:
        cov, tot, res = coverage(fix, prof)
        print(f"{name:20s} covered {cov}/{tot} ({100*cov/tot:.1f}%)  "
              f"projected residual wave {res:.1f} ms  (saved {total_wave-res:.1f} ms)")
    for p, n in used_cals:
        print(f"  calibration source: {p} ({n} rounds)")


if __name__ == "__main__":
    main()
