#!/usr/bin/env python3
"""Measure direct-model time to tested code, with isolated sequential servers.

Run on the benchmark machine with other inference/build workloads stopped.
Generated Python is checked by a restricted AST interpreter in a bounded child;
this harness never writes it into a user project or gives it shell tools.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request

from lib.qwen_coding_fixtures import load_cases, validate_response

PROFILES = {
    'default': {},
    'v3': {'CAMELID_KQUANT_V3': '1'},
    'v4': {'CAMELID_KQUANT_V4': '1', 'CAMELID_METAL_ATTN_BATCH_K': '1',
           'CAMELID_SPEC_VERIFY_BATCH_GLUE': '1'},
    'qwenmm': {'CAMELID_KQUANT_V3': '1', 'CAMELID_METAL_KQUANT_ATTN_MM': '1',
               'CAMELID_METAL_QK_NORM_ATTN_MM': '1'},
}
PROFILES['v4suffix'] = dict(PROFILES['v4'], CAMELID_SPEC_DECODE='suffix',
                           CAMELID_SPEC_GPU='1', CAMELID_SPEC_DRAFT_TOKENS='15')
PROFILES['qwenreuse'] = dict(PROFILES['qwenmm'], CAMELID_QWEN_PREFIX_REUSE='1',
                            CAMELID_QWEN_PREFIX_TRACE='1')
REFERENCES = {'v4suffix': 'v4', 'qwenreuse': 'qwenmm'}
FOLLOWUP = '\n\nReview all boundary cases once more. Return only the complete function with the same signature and required behavior.'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def health(base):
    with urllib.request.urlopen(base + '/v1/health', timeout=3) as response:
        return json.load(response)


def request_completion(base, payload, timeout):
    request = urllib.request.Request(base + '/v1/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        response = json.load(reply)
    return response, time.monotonic() - started


def run(args):
    selected = args.profiles.split(',')
    if len(set(selected)) != len(selected) or any(name not in PROFILES for name in selected):
        raise ValueError('Select unique known profiles: ' + ','.join(PROFILES))
    for port in {18191, 18192, 18193, args.port}:
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'Port {port} is occupied; serialize model workloads first')
    cases = load_cases(args.pack)
    if args.cases:
        wanted = args.cases.split(',')
        available = {c['id']: c for c in cases}
        cases = [available[name] for name in wanted]
    if not cases:
        raise ValueError('No cases selected')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 'camelid.direct-coding-benchmark/v1',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'host': socket.gethostname(), 'machine': os.uname().machine,
        'binary_sha256': digest(args.binary), 'model_sha256': digest(args.model),
        'pack_sha256': digest(args.pack), 'harness_sha256': digest(__file__),
        'validator_sha256': digest(Path(__file__).parent / 'lib/qwen_coding_fixtures.py'),
        'case_ids': [c['id'] for c in cases], 'warm_followup': args.warm_followup,
        'temperature': 0, 'tools': False}
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in manifest:
            if key not in ('created_utc',) and previous[key] != manifest[key]:
                raise ValueError(f'Existing experiment differs in {key}; use a fresh output directory')
    else:
        save(manifest_path, manifest)
    base = f'http://127.0.0.1:{args.port}'
    for profile in selected:
        arm = out / profile
        arm.mkdir(exist_ok=False)
        changes = dict(PROFILES[profile])
        env = {k: v for k, v in os.environ.items() if not k.startswith('CAMELID_')}
        env.update(changes, CAMELID_WORKSPACE_MEMORY_DB=str(arm / 'memory.sqlite3'), RUST_LOG='info')
        command = [str(Path(args.binary).resolve()), 'serve', '--addr', f'127.0.0.1:{args.port}',
                   '--model', str(Path(args.model).resolve()), '--max-prompt-tokens', '8192',
                   '--max-generation-tokens', str(max(c['max_tokens'] for c in cases)), '--no-open']
        save(arm / 'config.json', {'command': command, 'env': changes})
        with (arm / 'server.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            try:
                for _ in range(90):
                    if process.poll() is not None:
                        raise RuntimeError('Server exited before ready')
                    try:
                        status = health(base)
                        if status.get('generation_ready'):
                            break
                    except Exception:
                        pass
                    time.sleep(1)
                else:
                    raise TimeoutError('Engine did not become ready')
                for case in cases:
                    variants = [('cold', case['messages'])]
                    if args.warm_followup and case.get('context_target_tokens', 0) >= 3000:
                        messages = [dict(m) for m in case['messages']]
                        messages[-1]['content'] += FOLLOWUP
                        variants.append(('extension', messages))
                    for variant, messages in variants:
                        payload = {'model': status['active_model_id'], 'messages': messages,
                                   'temperature': 0, 'max_tokens': case['max_tokens'], 'stream': False}
                        offset = (arm / 'server.log').stat().st_size
                        response, wall = request_completion(base, payload, args.timeout)
                        choice = response['choices'][0]
                        text = choice['message'].get('content') or ''
                        validation = validate_response(case, text, choice['finish_reason'])
                        timing = response['camelid']['timings_ms']
                        prompt = timing['prompt_evaluation']
                        prompt_ms = prompt['prefill']['forward_total'] + prompt['first_token']['forward_total']
                        decode_ms = timing['generate'] - prompt_ms
                        usage = response['usage']
                        summary = {'profile': profile, 'case': case['id'], 'variant': variant,
                            'wall_seconds': wall, 'prompt_ms': prompt_ms, 'decode_ms': decode_ms,
                            'prompt_tokens': usage['prompt_tokens'], 'output_tokens': usage['completion_tokens'],
                            'decode_tokens_per_second': (usage['completion_tokens'] - 1) * 1000 / decode_ms if decode_ms > 0 else None,
                            'finish_reason': choice['finish_reason'], 'prompt_cache_hit': timing['prompt_cache_hit'],
                            'passed': validation['passed'],
                            'time_to_tested_code_seconds': wall + validation['duration_ms'] / 1000 if validation['passed'] else None}
                        with (arm / 'server.log').open('rb') as records:
                            records.seek(offset)
                            trace = records.read().decode(errors='replace')
                        reused = re.findall(r'prefix continuation: reused (\d+) of (\d+) positions, prefilled (\d+)', trace)
                        summary['gpu_prefix_reuse'] = [dict(zip(('reused', 'total', 'prefilled'), map(int, row))) for row in reused]
                        name = case['id'] + '-' + variant + '.json'
                        reference = out / REFERENCES.get(profile, '__none__') / name
                        if reference.exists():
                            plain = json.loads(reference.read_text())
                            if plain['request'] != payload:
                                raise ValueError('Same-kernel reference has different request')
                            expected = plain['response']['camelid']['generated_token_ids']
                            actual = response['camelid']['generated_token_ids']
                            summary['same_kernel_ids_match'] = expected == actual
                        save(arm / name, {'summary': summary, 'validation': validation,
                             'request': payload, 'response': response, 'health': status})
                        print(json.dumps(summary), flush=True)
                records = [json.loads(p.read_text())['summary'] for p in arm.glob('*.json')
                           if p.name not in ('config.json', 'error.json', 'score.json')]
                save(arm / 'score.json', {'passed': sum(r['passed'] for r in records), 'total': len(records),
                    'failed': [r['case'] + ':' + r['variant'] for r in records if not r['passed']],
                    'parity_failures': [r['case'] + ':' + r['variant'] for r in records if r.get('same_kernel_ids_match') is False]})
            except Exception as error:
                save(arm / 'error.json', {'type': type(error).__name__, 'message': str(error)})
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--pack', default=str(Path(__file__).parent.parent / 'qa/prompt-packs/qwen4b-coding-v1.json'))
    p.add_argument('--out', required=True)
    p.add_argument('--profiles', default='default,v3,v4,v4suffix')
    p.add_argument('--cases')
    p.add_argument('--warm-followup', action='store_true')
    p.add_argument('--port', type=int, default=18194)
    p.add_argument('--timeout', type=int, default=240)
    run(p.parse_args())
