#!/usr/bin/env python3
"""Summarize one bench run dir: tok/s, alpha, round split, correctness vs fixture."""
import json, re, sys, os

_HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED = json.load(
    open(os.path.join(_HERE, os.pardir, 'hybrid-hot48-runner', 'expected-48-token-ids.json'))
)

def summarize(d):
    log = open(os.path.join(d, 'server.log'), errors='replace').read()
    out = {'run': os.path.basename(d)}
    try:
        out['load_s'] = round(float(open(os.path.join(d,'timings.txt')).read().split('=')[1]), 2)
    except Exception: pass
    try:
        out['http_wall_s'] = round(float(open(os.path.join(d,'http-wall-seconds.txt')).read().strip()), 3)
    except Exception: pass

    rounds = [(float(a), float(b), float(c), int(e), int(f))
              for a,b,c,e,f in re.findall(
                  r'\[mtp round\] #\d+ wall=([\d.]+)ms \(assistant=([\d.]+)ms, verifier=([\d.]+)ms\) accepted=(\d+)/(\d+)', log)]
    if rounds:
        n = len(rounds)
        out['mtp_rounds'] = n
        out['round_wall_ms_median'] = round(sorted(r[0] for r in rounds)[n//2], 1)
        out['assistant_ms_median'] = round(sorted(r[1] for r in rounds)[n//2], 1)
        out['verifier_ms_median'] = round(sorted(r[2] for r in rounds)[n//2], 1)
        acc = [r[3] for r in rounds]
        out['accepted_mean'] = round(sum(acc)/n, 2)
        out['alpha_tokens_per_round'] = round(sum(acc)/n + 1, 2)
        out['decode_tok_s'] = round(1000.0 * (sum(acc)/n + 1) / (sum(r[0] for r in rounds)/n), 2)

    # per-round ledger (decode rounds only = those after the last prefill chunk)
    led = re.findall(r'\[metal chained ledger\] start_pos=(\d+) K=(\d+).*?final_wait=([\d.]+)ms encode=([\d.]+)ms gpu_busy\(last_cb\)=([\d.]+)ms disk_loads=(\d+) disk_bytes=([\d.]+)MiB disk_time=([\d.]+)ms', log)
    if led:
        dec = led[-len(rounds):] if rounds else led[-5:]
        f = lambda i: round(sum(float(x[i]) for x in dec)/len(dec), 1)
        out['ledger_decode'] = {'final_wait_ms': f(2), 'encode_ms': f(3), 'gpu_busy_last_cb_ms': f(4),
                                'disk_MiB': f(6), 'disk_ms': f(7),
                                'disk_GBps': round(sum(float(x[6]) for x in dec)/max(sum(float(x[7]) for x in dec),1e-9)/1024*1000, 2)}
    st = re.findall(r'\[metal chained stages\] split=\S+ qkv_o=([\d.]+)ms attn=([\d.]+)ms router=([\d.]+)ms shared=([\d.]+)ms gateup=([\d.]+)ms down=([\d.]+)ms resid=([\d.]+)ms gpu_total=([\d.]+)ms', log)
    if st:
        dec = st[-len(rounds):] if rounds else st[-5:]
        keys = ['qkv_o','attn','router','shared','gateup','down','resid','gpu_total']
        out['gpu_stages_ms'] = {k: round(sum(float(x[i]) for x in dec)/len(dec),1) for i,k in enumerate(keys)}
    idle = re.findall(r'slot hits=(\d+) misses=(\d+) evictions=(\d+)', log)
    if idle:
        dec = idle[-len(rounds):] if rounds else idle[-5:]
        h = sum(int(x[0]) for x in dec); m = sum(int(x[1]) for x in dec)
        out['slot_hit_rate'] = round(h/max(h+m,1), 3)
        out['misses_per_round'] = round(m/len(dec), 1)

    rp = os.path.join(d,'response.json')
    if os.path.exists(rp) and os.path.getsize(rp) > 0:
        try:
            r = json.load(open(rp))
            ids = r.get('camelid',{}).get('generated_token_ids')
            txt = r['choices'][0]['message']['content']
            out['completion_tokens'] = r.get('usage',{}).get('completion_tokens')
            out['text_head'] = txt[:110]
            if ids:
                pref = 0
                for a,b in zip(ids, EXPECTED):
                    if a!=b: break
                    pref += 1
                out['exact_match_expected'] = (ids == EXPECTED)
                out['exact_prefix_len'] = pref
        except Exception as e:
            out['response_error'] = str(e)[:120]
    return out

for d in sys.argv[1:]:
    print(json.dumps(summarize(d), indent=1))
