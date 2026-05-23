# FFN-Down Decode Chunking Plan Guard

Cron lane: `0719640b-b612-42e5-a335-f8f5d87fd3e7`

Source head at slice start: `66213ac`

## Result

Retained as a default-off control-plane slice only.

The runtime already had `CAMELID_X86_Q8_FFN_DOWN_DECODE_GROUP_CHUNKING` coverage, but the execution planner did not manage the flag with the rest of the Ubuntu x86 Q8 experimental gates. This slice adds planner ownership so stale values are cleared by default, explicit experimental opt-ins are preserved, and the public configuration/performance notes match the runtime route.

## Validation status

Local shell gate:

```text
uname -sm
Darwin arm64
```

Canonical Ubuntu validation attempt:

```text
ssh: connect to canonical Ubuntu validation host port 22: Operation timed out
rc=255
```

Because the control shell was not `Linux x86_64` and the canonical Ubuntu SSH probe timed out, no local cargo check, local benchmark, llama.cpp comparison, or cross-target result is claimed for this slice. The cron handoff for this run carries the exact stderr; this repository note avoids publishing private host locator details.

## Evidence boundary

This is planner/runtime-gate hygiene only. It does not promote support, throughput, RSS, default-on behavior, portability, or same-host llama.cpp parity.
