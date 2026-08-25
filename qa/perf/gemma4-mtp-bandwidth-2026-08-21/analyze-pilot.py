#!/usr/bin/env python3
"""Summarize a NIM pilot report + watchdog into the facts that matter."""
import json, sys, os

operator_home = os.environ.get("CAMELID_OPERATOR_HOME") or os.environ.get("HOME")
if not operator_home:
    raise RuntimeError("set CAMELID_OPERATOR_HOME or HOME")
model_root = os.environ.get("CAMELID_MODEL_ROOT") or os.path.join(operator_home, "models")
D = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.environ.get("CAMELID_NIM_PILOT_RUN_DIR")
    or os.path.join(model_root, "gemma4-mtp-pair", "runs", "mtp-nim-table32.qHR2Wmcz")
)
rep = os.path.join(D, 'nim-pilot-report.json')
wd  = os.path.join(D, 'nim-watchdog.jsonl')

if os.path.exists(wd):
    rows = [json.loads(l) for l in open(wd)]
    base = next((r for r in rows if r['event'] == 'clean_parent_baseline'), None)
    fin  = next((r for r in rows if r['event'] == 'final'), None)
    ab   = [r for r in rows if r['event'] in ('watchdog_abort', 'baseline_refused')]
    samples = [r for r in rows if r['event'] == 'sample']
    print('=== WATCHDOG ===')
    if base:
        b = base['host']
        print(f"  baseline: headroom {b['reclaimable_headroom_bytes']/1e9:.2f} GB  free {b['free_bytes_strict']/1e9:.2f} GB  swapouts {b['swapouts_pages']}")
    print(f"  samples: {len(samples)}  (~{len(samples)*0.25:.1f}s observed)")
    if samples:
        hs = [s['host'] for s in samples]
        print(f"  headroom min {min(h['reclaimable_headroom_bytes'] for h in hs)/1e9:.2f} GB  "
              f"wired max {max(h['wired_bytes'] for h in hs)/1e9:.2f} GB  "
              f"pressure levels {sorted(set(h['pressure_level_raw'] for h in hs))}")
    for a in ab:
        print(f"  !! {a['event']}: {a.get('violations')}")
    if fin:
        print(f"  final: rc={fin['child_returncode']} aborted={fin['watchdog_aborted']} reasons={fin['abort_reasons']}")
        print(f"         min headroom {fin['minimum_reclaimable_headroom_bytes_observed']/1e9:.2f} GB")
else:
    print('=== WATCHDOG === (no watchdog log yet)')

if not os.path.exists(rep):
    print('\n=== REPORT === (not written)')
    sys.exit(0)

r = json.load(open(rep))
print('\n=== REPORT ===')
print(f"  pilot_only={r.get('pilot_only')} kill_reason={json.dumps(r.get('kill_reason'))}")
print(f"  runs={len(r.get('runs',[]))} economics={len(r.get('economics',[]))} lane_memory={len(r.get('lane_memory',[]))}")
print(f"  unverified_assistant_tokens_committed={r.get('unverified_assistant_tokens_committed')} target_authoritative={r.get('target_authoritative')}")
print(f"  incorporated_child_receipts={r.get('incorporated_child_receipts')}")

for run in r.get('runs', []):
    print(f"\n  -- lane={run.get('lane')} phase={run.get('phase')} rep={run.get('repetition')} completed={run.get('completed')}")
    print(f"     wall_us={run.get('generation_wall_us')} emitted={run.get('emitted_tokens')} rounds={len(run.get('rounds',[]))}")
    if run.get('failure'): print(f"     FAILURE: {run['failure']}")
    snap = run.get('routed_expert_residency') or {}
    if snap:
        print(f"     unique/layer max={snap.get('interval_routed_unique_experts_max')} sum={snap.get('interval_routed_unique_experts_sum')}")
        print(f"     last_chained_unique_max={snap.get('last_chained_unique_experts_max')} per_layer={snap.get('last_chained_unique_per_layer')}")
        print(f"     physical_base_slot_budget={snap.get('physical_base_slot_budget')} occupied={snap.get('occupied_base_slots')}")
        print(f"     capacity_overflow={snap.get('last_chained_slot_capacity_overflow')} dropped={snap.get('last_chained_selected_experts_dropped')} missing={snap.get('last_chained_missing_expert_failclose')}")

for e in r.get('economics', []):
    print(f"\n  ECONOMICS {e.get('workload')} rep{e.get('repetition')}: {json.dumps(e)[:400]}")
