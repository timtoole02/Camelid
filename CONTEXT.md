# Camelid Context

A shared language for Camelid agents and contributors. Keep this file as a glossary, not a spec or scratchpad.

## Language

**Camelid**
The product: a local-first LLM runtime and chat surface. Use this name in public-facing product language.
_Avoid_: backendinference except when referring to historical crate/repo names.

**Support contract**
The public truth about which exact model rows and workflows are supported. A row is supported only when parity evidence, API behavior, frontend behavior, and docs agree.
_Avoid_: “works” without row-specific evidence.

**Evidence bundle**
A durable directory containing commands, raw logs/JSON, summaries, commit SHA, and dirty diff/stat for a validation or performance claim.
_Avoid_: screenshots or chat summaries as primary proof.

**Retained slice**
A default-off or default-on implementation slice that passed the retain bar against the current baseline and has an evidence bundle.
_Avoid_: “candidate” once a slice is retained.

**Rejected slice**
An attempted implementation with preserved evidence explaining why it lost, regressed, or remained inconclusive. Rejected slices prevent repeated dead ends.
_Avoid_: deleting failed experiments without a note.

**Current retained baseline**
The latest retained implementation/evidence pair for a lane. New performance work must compare against this, not an older weaker baseline.
_Avoid_: old-baseline wins.

**Same-host guard**
A comparison run where Camelid and the reference engine execute on the same machine under comparable conditions.
_Avoid_: cross-machine performance claims.

**Canonical Ubuntu host report**
A validation status statement about the canonical Ubuntu host is current only when the project-private validation probe was attempted in the same run and its stderr is cited on failure. If that probe was not attempted in the current run, say remote validation was not attempted instead of implying a negative host reachability or authentication status.
_Avoid_: publishing private hostnames, key paths, usernames, IP addresses, or stale host-status wording copied from older summaries.

**Tracer bullet**
A narrow vertical slice that proves or rejects one idea end-to-end through the real path.
_Avoid_: horizontal rewrites that touch many layers without a fast retain/reject gate.

**Deep module**
A module with a small interface hiding substantial behavior and invariants. Q8 packing, scheduling, and tensor runtime storage should move toward deep modules.
_Avoid_: shallow pass-through wrappers.

**Backend-owned packed runtime storage**
The target architecture where packed Q8 runtime layout is owned by the backend/tensor runtime and consumed directly by hot linear ops.
_Avoid_: duplicate sidecar packed copies as the final performance architecture.

**Q8 projection route resolver**
An internal fail-closed decision seam that validates whether one dense Q8 projection family may use backend-owned packed runtime storage for a specific decode or prefill policy. It should consume the resolved runtime plan, packed-storage invariants, row policy, and dimensions, then hand already-validated storage to math helpers.
_Avoid_: treating a resolver cleanup as throughput, RSS, profiling, support-contract, frontend, or default-on evidence.

**Parity envelope**
The exact model, quantization, prompt, context, token count, and sampling settings where Camelid has proven equality or bounded equivalence.
_Avoid_: extending a parity result beyond its envelope.

## Relationships

- A **support contract** is justified by one or more **evidence bundles**.
- A **retained slice** updates the **current retained baseline** for its lane.
- A **rejected slice** documents a dead end and shapes the next **tracer bullet**.
- A **same-host guard** is one feedback loop used to validate a **parity envelope** or performance claim.
- **Backend-owned packed runtime storage** is a desired **deep module** direction for Q8 acceleration.
- A **Q8 projection route resolver** is one way to make backend-owned packed runtime storage a deeper module; it can support a **retained slice** only inside the exact evidence envelope it was gated against.

## Flagged ambiguities

- “Mac Q8” means native Apple Silicon/macOS arm64 CPU work unless a document explicitly says otherwise. It does not mean Metal.
- “Ubuntu Q8” means Ubuntu x86_64 work and should not borrow Mac evidence.
- “Supported” means support-contract aligned, not merely “generated one token once.”
