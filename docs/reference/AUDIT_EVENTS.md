# Agent audit events

The agent tool loop (`camelid chat --agent`, and the `agent-eval` harness) emits
a pair of audit events around **every tool that actually executes**:

- `agent.tool_call` — immediately **before** the tool runs.
- `agent.tool_result` — immediately **after** it returns.

A tool that is denied (by the approval policy's `deny` tier or a user `no`), or
that fails sandbox validation, **does not execute** and therefore emits **no**
events. So a clean run produces exactly one `tool_call` + one `tool_result` per
executed tool.

Events are produced in `src/chat/audit.rs` and delivered through a pluggable
[`AuditSink`]. The agent loop calls `sink.emit(...)` and never blocks on it.

## Sinks

| Sink           | Behavior                                                                 |
| -------------- | ------------------------------------------------------------------------ |
| `NoopSink`     | **Default.** Emits nothing. An unconfigured deployment audits nothing and never errors. |
| `WebhookSink`  | POSTs each event as JSON to a configured URL on a background thread. **Non-blocking**: events go through a bounded queue (capacity 256); if the worker falls behind and the queue is full, events are **dropped** rather than stalling the agent loop. |
| `InMemorySink` | Collects events in memory (tests / in-process consumers).                |

No endpoint is hardcoded. The webhook URL is supplied by, in precedence order:

1. `camelid chat --agent --audit-webhook <URL>`
2. the `CAMELID_AUDIT_WEBHOOK` environment variable
3. otherwise unset → `NoopSink` (no audit)

The webhook worker shells out to `curl` (already a runtime dependency) with a
5-second per-request timeout; the JSON body is piped via stdin, so it never
appears in the process table. If `curl` is missing the event is silently skipped
— audit delivery must never crash the agent.

## Event schema

Both events share one JSON shape. `outcome` and `duration_ms` are `null` on
`agent.tool_call` (the tool has not run yet) and populated on
`agent.tool_result`.

```json
{
  "event": "agent.tool_call",        // or "agent.tool_result"
  "timestamp_unix_ms": 1718553600123, // event creation time, ms since epoch
  "tool": "run_shell",               // tool name
  "approval_tier": "confirm",        // tier applied: "auto" | "confirm" | "deny"
  "args_digest": "sha256:9f86d0…",   // SHA-256 of the canonical args JSON
  "outcome": "ok",                   // "ok" | "error" | null (null on tool_call)
  "duration_ms": 12                  // execution wall time | null (null on tool_call)
}
```

### Field notes

- **`args_digest`** — a `sha256:<hex>` digest of the tool's arguments, **not the
  arguments themselves**. Tool inputs can contain secrets (tokens, paths,
  command strings); the audit trail records a stable hash so two events can be
  correlated and identical calls recognized, without ever transmitting the raw
  input. The hash is taken over `serde_json`'s canonical (key-sorted) encoding,
  so it is deterministic for equal arguments.
- **`approval_tier`** — the tier the policy applied to this call (see Task 2 /
  `ApprovalPolicy`). Because only executed tools emit events, this is `auto`,
  `confirm` (then approved), or an explicit override — never a `deny`.
- **`outcome`** — `error` reflects a tool-level error result (e.g. a failed
  shell command, a read error), not a transport error. The raw output text is
  deliberately **not** included (it, too, can carry secrets or injected
  content).
- **`timestamp_unix_ms`** — wall-clock at event construction. The
  `tool_result` timestamp minus the `tool_call` timestamp will be close to, but
  is not authoritative for, `duration_ms`; use `duration_ms` for timing.

## Correlating a call with its result

Within one agent loop, tools execute serially, so a `tool_call` is always
immediately followed by its `tool_result`. For out-of-band correlation, match on
`(tool, args_digest)` and the adjacent timestamps. (A dedicated correlation id
can be added if a consumer needs to interleave across concurrent agents.)
