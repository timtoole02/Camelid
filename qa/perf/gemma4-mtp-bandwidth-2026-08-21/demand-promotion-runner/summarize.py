#!/usr/bin/env python3
"""Summarize one bench run dir: tok/s, alpha, round split, correctness vs fixture."""
import base64, binascii, json, re, sys, os

_HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED = json.load(
    open(os.path.join(_HERE, os.pardir, 'hybrid-hot48-runner', 'expected-48-token-ids.json'))
)
ENV_KEY_RE = r'[A-Za-z_][A-Za-z0-9_]*'


def read_env_manifest(path):
    """Read legacy KEY=VALUE receipts and strict base64-v1 manifests."""
    lines = open(path, errors='strict').read().splitlines()
    meaningful = [line for line in lines if line]
    strict = bool(meaningful and meaningful[0] == 'manifest_format=base64-v1')
    values = {}
    for line in meaningful:
        if line == 'manifest_format=base64-v1':
            key, value = 'manifest_format', 'base64-v1'
        else:
            encoded = re.fullmatch(rf'({ENV_KEY_RE})@BASE64=([A-Za-z0-9+/]*={{0,2}})', line)
            if encoded:
                key = encoded.group(1)
                try:
                    value = base64.b64decode(encoded.group(2), validate=True).decode('utf-8')
                except (binascii.Error, UnicodeDecodeError) as error:
                    raise ValueError(f'invalid base64 manifest value for {key}') from error
            else:
                plain = re.fullmatch(rf'({ENV_KEY_RE})=(.*)', line)
                if strict or plain is None:
                    if strict:
                        raise ValueError(f'malformed base64-v1 manifest line: {line!r}')
                    # Historical KEY@FILE receipts expanded multiline values
                    # directly. Ignore their continuation lines while retaining
                    # exact-key parsing for the ordinary environment entries.
                    continue
                key, value = plain.groups()
        if key in values:
            raise ValueError(f'duplicate environment manifest field {key}')
        values[key] = value
    return values

def summarize(d):
    log = open(os.path.join(d, 'server.log'), errors='replace').read()
    out = {'run': os.path.basename(d)}
    try:
        env = read_env_manifest(os.path.join(d, 'env.txt'))
        if env.get('CAMELID_BENCH_CEILING_ONLY') == 'oracle-seeded-ngram':
            out['ceiling_only'] = True
            out['draft_source'] = 'oracle-seeded-ngram'
            out['throughput_promotion_allowed'] = False
    except Exception as error:
        out['env_manifest_error'] = str(error)[:120]
    try:
        out['load_s'] = round(float(open(os.path.join(d,'timings.txt')).read().split('=')[1]), 2)
    except Exception: pass
    try:
        out['http_wall_s'] = round(float(open(os.path.join(d,'http-wall-seconds.txt')).read().strip()), 3)
    except Exception: pass

    rounds = [(float(a), float(b), float(c), int(e), int(f))
              for a,b,c,e,f in re.findall(
                  r'\[mtp round\] #\d+ wall=([\d.]+)ms \(assistant=([\d.]+)ms, verifier=([\d.]+)ms\) accepted=(\d+)/(\d+)', log)]
    round_kind = 'mtp'
    if not rounds:
        rounds = [(float(a), float(b), float(c), int(e), int(f))
                  for a,b,c,e,f in re.findall(
                      r'\[spec round\] #\d+ wall=([\d.]+)ms \(draft=([\d.]+)ms, verifier=([\d.]+)ms\) accepted=(\d+)/(\d+)', log)]
        round_kind = 'spec'
    if rounds:
        n = len(rounds)
        out[f'{round_kind}_rounds'] = n
        out['round_wall_ms_median'] = round(sorted(r[0] for r in rounds)[n//2], 1)
        out[f'{"assistant" if round_kind == "mtp" else "draft"}_ms_median'] = round(sorted(r[1] for r in rounds)[n//2], 1)
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

def main(argv=None):
    paths = sys.argv[1:] if argv is None else argv
    for path in paths:
        print(json.dumps(summarize(path), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
