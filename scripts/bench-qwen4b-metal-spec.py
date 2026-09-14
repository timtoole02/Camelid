#!/usr/bin/env python3
"""Sequential, isolated Qwen4B Metal/chat experiment.

Run on the benchmark host with all other model workloads stopped. The harness
refuses occupied preview/comparison ports and owns only its child server. Artifacts include
the request, output token IDs, binary/model hashes, environment and full server log.
No candidate is promoted by this script. Compare speculative IDs to the SAME
kernel's plain IDs: arithmetic variants are separate correctness baselines.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

CASES = {
    'prose': 'Explain how snow forms in about 150 words, using simple language.',
    'code': 'Write a Python function that merges two sorted lists into one sorted list without using sorted(). Include a concise explanation and three assert tests.',
    'copy': 'Copy the following records exactly, without a code fence or explanation:\n' + '\n'.join(f'item_{i:02d},active,{i * 7}' for i in range(35)),
    'long': 'Reference notes:\n' + 'The garden has trees, flowers, birds, and a small pond. ' * 100 + '\nSummarize these notes in about 100 words.',
}
CASES['deepcopy'] = 'Background notes (do not repeat):\n' + ('The garden has trees, flowers, birds, and a small pond. ' * 270) + '\nEnd of notes.\n' + CASES['copy']
ARMS = {
    'v1': {},
    'v2': {'CAMELID_KQUANT_V2': '1'},
    'v3': {'CAMELID_KQUANT_V3': '1'},
    'v4': {'CAMELID_KQUANT_V4': '1'},
    'v4batch': {'CAMELID_KQUANT_V4': '1', 'CAMELID_METAL_ATTN_BATCH_K': '1'},
}
ARMS['v4glue'] = dict(ARMS['v4batch'], CAMELID_SPEC_VERIFY_BATCH_GLUE='1')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def get(base, path):
    return json.load(urllib.request.urlopen(base + path, timeout=3))

def run(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = f'http://127.0.0.1:{args.port}'
    for port in (18191, 18192, 18193, args.port):
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'Port {port} is occupied; stop model servers before benchmarking')
    metadata = {'binary': str(Path(args.binary).resolve()), 'binary_sha256': sha(args.binary),
                'model': str(Path(args.model).resolve()), 'model_sha256': sha(args.model),
                'host': socket.gethostname(), 'machine': os.uname().machine,
                'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'max_tokens': args.max_tokens, 'cases': CASES, 'stream': args.stream}
    manifest = out / 'manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        for key in ('binary_sha256', 'model_sha256', 'max_tokens', 'cases', 'stream'):
            if previous[key] != metadata[key]:
                raise ValueError(f'Existing experiment differs in {key}; use a new output directory')
    else:
        manifest.write_text(json.dumps(metadata, indent=2))
    for label in args.arms.split(','):
        parts = label.split('-')
        env_changes = dict(ARMS[parts[0]])
        if len(parts) > 1:
            if len(parts) != 3 or parts[1] not in ('ngram', 'suffix', 'draft') or not 1 <= int(parts[2]) <= 15:
                raise ValueError('Speculative arm must be KERNEL-{ngram,suffix,draft}-DRAFTS with DRAFTS in 1..15')
            env_changes.update(CAMELID_SPEC_DECODE=parts[1], CAMELID_SPEC_GPU='1',
                               CAMELID_SPEC_DRAFT_TOKENS=parts[2])
            if parts[1] == 'draft':
                if not args.draft_model:
                    raise ValueError('--draft-model required')
                env_changes['CAMELID_SPEC_DRAFT_MODEL'] = args.draft_model
        arm = out / label
        arm.mkdir(exist_ok=False)
        env = {k: v for k, v in os.environ.items() if not k.startswith('CAMELID_')}
        if args.stream:
            env_changes['CAMELID_STREAM_TIMING_DIAGNOSTICS'] = '1'
        env.update(env_changes, CAMELID_WORKSPACE_MEMORY_DB=str(arm / 'memory.sqlite3'), RUST_LOG='info')
        cmd = [args.binary, 'serve', '--addr', f'127.0.0.1:{args.port}', '--model', args.model,
               '--max-prompt-tokens', '8192', '--max-generation-tokens', str(args.max_tokens), '--no-open']
        (arm / 'config.json').write_text(json.dumps({'cmd': cmd, 'env': env_changes,
            'draft_model_sha256': sha(args.draft_model) if len(parts) > 1 and parts[1] == 'draft' else None}, indent=2))
        with (arm / 'server.log').open('w') as log:
            proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            try:
                for _ in range(90):
                    if proc.poll() is not None:
                        raise RuntimeError(f'{label} server exited; see log')
                    try:
                        health = get(base, '/v1/health')
                        if health.get('generation_ready'):
                            break
                    except Exception:
                        pass
                    time.sleep(1)
                else:
                    raise TimeoutError('Engine did not become ready')
                for name in args.cases.split(','):
                    payload = {'model': health['active_model_id'], 'messages': [{'role': 'user', 'content': CASES[name]}],
                               'temperature': 0, 'max_tokens': args.max_tokens, 'stream': args.stream}
                    if args.stream:
                        payload['stream_options'] = {'include_usage': True}
                    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(payload).encode(),
                                                 headers={'Content-Type': 'application/json'})
                    started = time.monotonic()
                    stream_metrics = {}
                    if args.stream:
                        events, chunks, usage, diagnostics = [], [], None, None
                        first_content, done = None, False
                        with urllib.request.urlopen(req, timeout=240) as reply:
                            for raw in reply:
                                line = raw.decode().strip()
                                if not line.startswith('data:'):
                                    continue
                                data = line[5:].strip()
                                if data == '[DONE]':
                                    done = True
                                    break
                                event = json.loads(data)
                                events.append(event)
                                if event.get('error'):
                                    raise RuntimeError(f'Stream error: {event["error"]}')
                                usage = event.get('usage') or usage
                                diagnostics = event.get('camelid') or diagnostics
                                for choice in event.get('choices', []):
                                    chunk = choice.get('delta', {}).get('content')
                                    if chunk:
                                        if first_content is None:
                                            first_content = time.monotonic() - started
                                        chunks.append(chunk)
                        if not done or not usage or not diagnostics:
                            raise RuntimeError('Stream missing DONE, usage, or diagnostics')
                        t = diagnostics['stream_timing_diagnostics']['timings_ms']
                        prompt_ms = t['prefill_forward_total'] + t['first_token_forward_total']
                        response = {'usage': usage, 'events': events, 'content': ''.join(chunks)}
                        stream_metrics = {'first_content_seconds': first_content,
                                          'content_chunks': len(chunks), 'done': done}
                    else:
                        response = json.load(urllib.request.urlopen(req, timeout=240))
                        t = response['camelid']['timings_ms']
                        pe = t['prompt_evaluation']
                        prompt_ms = pe['prefill']['forward_total'] + pe['first_token']['forward_total']
                    wall = time.monotonic() - started
                    # Includes orchestration and drafting, not just the target GPU time.
                    decode_ms = t['generate'] - prompt_ms
                    count = response['usage']['completion_tokens']
                    summary = {'arm': label, 'case': name, 'wall_seconds': wall, 'prompt_ms': prompt_ms,
                               'decode_ms': decode_ms, 'tokens': count,
                               'decode_tokens_per_second': (count - 1) * 1000 / decode_ms if decode_ms > 0 else None,
                               'prompt_cache_hit': t['prompt_cache_hit'], **stream_metrics}
                    plain_file = out / parts[0] / (name + '.json')
                    if len(parts) > 1 and plain_file.exists():
                        plain_record = json.loads(plain_file.read_text())
                        if plain_record['request'] != payload:
                            raise ValueError('Plain reference has different request parameters')
                        if args.stream:
                            summary['same_kernel_plain_text_match'] = (
                                plain_record['response']['content'] == response['content'])
                            summary['same_kernel_plain_usage_match'] = (
                                plain_record['response']['usage'] == response['usage'])
                        else:
                            plain = plain_record['response']['camelid']['generated_token_ids']
                            actual = response['camelid']['generated_token_ids']
                            summary['same_kernel_plain_ids_match'] = plain == actual
                            summary['first_divergence'] = next((i for i, (a, b) in enumerate(zip(plain, actual)) if a != b),
                                                               -1 if len(plain) == len(actual) else min(len(plain), len(actual)))
                    (arm / (name + '.json')).write_text(json.dumps({'summary': summary, 'request': payload,
                                                                   'response': response, 'health': health}, indent=2))
                    print(json.dumps(summary), flush=True)
                    for key in ('same_kernel_plain_ids_match', 'same_kernel_plain_text_match', 'same_kernel_plain_usage_match'):
                        if summary.get(key) is False:
                            raise RuntimeError(f'{label}/{name}: {key} failed; see saved receipt')
            except Exception as error:
                (arm / 'error.json').write_text(json.dumps({'type': type(error).__name__,
                                                           'message': str(error)}, indent=2))
                raise
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--draft-model')
    p.add_argument('--out', required=True)
    p.add_argument('--port', type=int, default=18194)
    p.add_argument('--arms', default='v1,v2,v3,v4')
    p.add_argument('--cases', default='prose,code,copy,long')
    p.add_argument('--stream', action='store_true', help='Measure actual SSE, comparing text and usage (no token IDs in SSE)')
    p.add_argument('--max-tokens', type=int, default=128)
    args = p.parse_args()
    if not 2 <= args.max_tokens <= 2048:
        p.error('--max-tokens must be in 2..2048')
    run(args)
