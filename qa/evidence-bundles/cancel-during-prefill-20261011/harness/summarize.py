"""Print the README's result table from data/api/*.jsonl."""
import json
import pathlib

ROWS = [
    ("Llama 3.2 3B Q4_K_M", "GPU (default)", "llama32-3b-gpu-main", "llama32-3b-gpu-fix"),
    ("Llama 3.2 3B Q4_K_M", "CPU (`--gpu off`)", "llama32-3b-cpu-main", "llama32-3b-cpu-fix"),
    ("Ornith 1.0 9B Q8_0", "CPU (default)", "ornith-9b-cpu-main", "ornith-9b-cpu-fix"),
    ("Ornith 1.0 9B Q8_0", "CUDA (`CAMELID_QWEN35_CUDA=1`)", "ornith-9b-cuda-main", "ornith-9b-cuda-fix"),
]
data = pathlib.Path(__file__).resolve().parent.parent / "data" / "api"


def load(name):
    return {row["check"]: row["r"] for row in map(json.loads, (data / f"{name}.jsonl").read_text().splitlines())}


def s(ms):
    return f"{ms / 1000:.1f} s"


print("| Model | Lane | Prompt tokens | Tiny reply, idle (main / PR) | Tiny reply after a cancel, main | Tiny reply after a cancel, this PR |")
print("|---|---|---|---|---|---|")
for model, lane, main, fix in ROWS:
    m, f = load(main), load(fix)
    tokens = m["stress-reference"]["reference"]["usage"]["prompt_tokens"]
    idle = f'{s(m["cancel-nonstream"]["idle"]["ms"])} / {s(f["cancel-nonstream"]["idle"]["ms"])}'
    after_m = [m[c]["after"]["ms"] for c in ("cancel-nonstream", "cancel-stream")]
    after_f = [f[c]["after"]["ms"] for c in ("cancel-nonstream", "cancel-stream")]
    print(f"| {model} | {lane} | {tokens} | {idle} | {s(after_m[0])} / {s(after_m[1])} | {s(after_f[0])} / {s(after_f[1])} |")

print()
print("| Lane | Build | Random cancels | Slowest tiny reply after one | Tiny reply text unchanged | Same-prompt reply equals uncancelled run |")
print("|---|---|---|---|---|---|")
stress = [
    ("Llama 3.2 3B GPU", "main", "llama32-3b-gpu-main"), ("Llama 3.2 3B GPU", "this PR", "llama32-3b-gpu-fix"),
    ("Llama 3.2 3B CPU", "main", "llama32-3b-cpu-main"), ("Llama 3.2 3B CPU", "this PR", "llama32-3b-cpu-fix"),
    ("Ornith 9B CPU", "main", "ornith-9b-cpu-main"), ("Ornith 9B CPU", "this PR (earlier build, see below)", "ornith-9b-cpu-fix-earlier-build"),
    ("Ornith 9B CUDA", "main", "ornith-9b-cuda-main"), ("Ornith 9B CUDA", "this PR", "ornith-9b-cuda-fix"),
]
for lane, build, name in stress:
    r = load(name)
    st = r["stress"]
    ref = r["stress-reference"]["reference"]["text"]
    same = st["final"]["text"] == ref and st["final"]["status"] == 200
    print(f"| {lane} | {build} | {len(st['trials'])} | {s(st['maxTinyMs'])} | {'yes' if st['allTinySame'] else 'NO'} | {'yes' if same else 'NO'} |")

for name in ("llama32-3b-gpu-main", "llama32-3b-gpu-fix", "llama32-3b-cpu-main", "llama32-3b-cpu-fix"):
    q = load(name)["queue"]
    print(f"queue {name}: C waited {s(q['c']['waitedMs'])}, finished {s(q['cEndMinusAEndMs'])} after A")
