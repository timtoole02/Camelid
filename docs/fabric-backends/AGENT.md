# Local Backend Fabric — agent handover

Read this before touching anything in the `fabric` lane. It states what the feature is for, what
is already committed, what is deliberately unfinished, and the testing standard the lane is held
to. If you are an agent picking this up, the **Testing standard** and **Traps** sections are the
parts that will actually stop you shipping something wrong.

---

## 1. End goal

**Camelid becomes the honest control plane for every local inference engine on a developer's
machines — including the ones it did not write.**

A developer who already runs Ollama or LM Studio should be able to point Camelid at them and get a
truthful answer to three questions:

1. **What have I actually got?** Which machines, which engines, which models, which of them can be
   served right now — and where the answer is unknown, *that it is unknown*, rather than a zero.
2. **What can each of them actually do?** Tool calls, embeddings, load reporting, warm prefixes —
   and **how we know**: measured here, declared by the API surface, or not checked.
3. **Do they agree with each other?** The same prompt on two engines, with a verdict that is
   allowed to say "nothing can be concluded", and — where the engine exposes it — the chat template
   that explains the difference.

The differentiator is not routing. Anyone can proxy to Ollama. The differentiator is that this tool
**tells you when your backends disagree, and why**, and refuses to state things it has not
established — including about itself.

### Why this matters commercially

Routing to a foreign engine makes us a worse version of that engine: we cannot see its queue, cannot
distinguish its refusals, cannot attest its cache. Telling a developer that their Ollama answers
differently from Camelid on identical weights, and showing them the two templates, is something no
one else does. The capability matrix and the divergence view are the product. Mixed routing is a
convenience built on top.

---

## 2. Parameters of success

This feature is successful when **all** of the following hold. They are deliberately falsifiable.

| # | Success parameter | How it is proven |
|---|---|---|
| S1 | A developer can add an existing Ollama/LM Studio install and see everything it holds, without editing code | live receipt against real engines |
| S2 | Nothing on screen or on the wire is fabricated. An unknown renders as an unknown | ablation: fabricate a zero, a named check must fail |
| S3 | A capability answer is never stronger than its evidence | ablation: credit an unmeasured backend, a named check must fail |
| S4 | The fabric never claims two backends agree, or disagree, without having established it | ablation: attribute a difference from an unstable side, a named check must fail |
| S5 | Default behaviour for existing users is byte-identical | every new feature is opt-in; "unaffected" tests exist alongside every guard |
| S6 | Adding a fourth engine is a new file plus a row, not a redesign | `policy.rs` contains no engine name; verified by grep each phase |
| S7 | Every claim in the docs has a receipt or is marked as not yet established | see "What is NOT claimed" in §4 |

**S6 is checked mechanically. Run this and expect no output:**

```bash
# Production code only. The test module below `#[cfg(test)]` legitimately names
# engines to build fixtures, so scanning the whole file gives a false alarm.
for f in policy discover identify netscope; do
  sed -n '1,/^#\[cfg(test)\]/p' "src/fabric/$f.rs" |
    grep -nE 'Ollama|LmStudio|NodeEngine::Camelid|"camelid"|"ollama"|"lmstudio"'
done
```

P4 extended this to the three discovery modules. They iterate `NodeEngine::ALL`
and never name a member: the identification paths, the signatures, the default
ports, the proxy-shape `NotANode` and `receives_fabric_bearer` all live in
`engine.rs` and the per-engine modules, so a fourth engine is still a new file
plus a row.

If that ever prints, the seam has leaked and placement has started knowing about engines by name.
Verified on this branch: production `policy.rs` is lines 1–787 and names no engine.

---

## 3. Invariants — do not violate these to make something work

These came out of real defects. Each one has caught a bug at least once.

- **I1** Default routing is Camelid-only. Mixed mode is explicit and opt-in.
- **I2** Every answer names the engine that served it.
- **I3** *Unknown is a value.* Never fabricate load, queue depth, capacity or readiness for a
  backend that does not report it. **A zero is a claim.**
- **I4** Affinity is **refused** for a node that cannot attest prefix warmth, never silently
  degraded.
- **I5** A foreign node is **not** tool-capable until measured on that backend *at that version*.
- **I6** No UI element without a backing field in a real response. No status derived from browser
  storage.
- **I7** No control that does not control. If we cannot start a remote process, render the command,
  not a button.
- **I8** Adding a node is always an explicit human action. Discovery proposes; a person disposes.
- **I9** We do not claim answer parity across engines. We measure divergence and display it.
- **I10** No throughput or latency claim without a fresh paired receipt on the exact head.
- **I11** Existing fail-closed transport rules are never loosened to reach a foreign backend.
- **I12** Placement reads exactly these parts of a request body and nothing else:
  - `model`, which it may write, and only `model`: to the id the chosen node uses under an
    operator-declared alias, or, under --allow-mixed-engines, to the chosen node's single active
    model when the request named none;
  - whether a top-level `tools` or `functions` key is present, never its contents;
  - in completion-time mode only, `stream`, `max_tokens`/`max_completion_tokens`, and the
    power-of-two bucket of the encoded body size.

  Message and tool content can influence placement only through that size bucket. The readers are
  `placement_requirements` and `service_class`.

Two more added by this work:

- **I13** A difference between two nodes is only attributable when **each node has been shown to
  agree with itself**. One run per side proves nothing.
- **I14** The fabric never infers that two model names mean the same weights. A human declares it,
  and the claim travels with every result that rests on it.

---

## 4. What is committed

All of the following is on this PR branch and green. Commits, newest first:

| Commit | One-liner |
|---|---|
| `7c096301` | P5 core — place on foreign engines behind `--allow-mixed-engines`, with four guards |
| `00450994` | O6 — an operator declares what each node calls a model (`alias` lines) |
| `9ff9f0a6` | Fix — the Compare screen had no nav entry; CI's own token-inspector smoke caught it |
| `a7a290b0` | Merge `upstream/main` (270 commits, v0.7.0 → v0.7.2) |
| `f8157606` | P0–P3 + P6 — engine seam, Ollama, LM Studio, capability matrix, divergence view |

### Phase by phase

| Phase | State | One-liner |
|---|---|---|
| **P0** Ground-truth lock | done | Baseline measured at `6618ffb7` before any change, so every later number has something to be compared against. |
| **P1** Honest fabric GUI | done | Deleted the sample-fabric generator, the storage-derived `live` chip and worker buttons that controlled nothing; the view now renders only fields present in a real response, and distinguishes *"this proxy withholds node detail"* from *"this fabric has no nodes"*. |
| **P2** Engine seam + Ollama | done | `LABEL=[ENGINE://]HOST[:PORT]`; engine **declared, never detected**; Ollama read via `/api/version`, `/api/tags`, `/api/ps`; reports **no load at all** rather than a zero. |
| **P3** LM Studio + capabilities | done | LM Studio read via `/api/v0/models` only; **no version endpoint exists**, so its version is unknown, not guessed; capability matrix with `measured` / `declared` / `not_probed` provenance, keyed on the exact version string. |
| **P6** Divergence view | done | `fabric compare`, `POST /v1/fabric/compare`, GUI Screen D. Runs each side N times, withholds a verdict unless both sides are self-consistent, suppresses the diff when nothing can be attributed, never names a winner. |
| **O6** Model identity | done | `alias CANONICAL=LABEL:LOCAL` in the nodes file. Ollama suffixes `:latest`, LM Studio does not, neither publishes a comparable digest — so a human declares it and it is recorded as `asserted_by_operator`. |
| **P5** Mixed routing | **core only** | Eligibility, ranking, tools and affinity guards + CLI flag + 9 tests. **Screen E and a live mixed receipt are NOT done.** |
| **P4** Discovery | done | `fabric discover` plus three proxy routes and a "Find machines" panel, sharing one implementation. Loopback-only by default; a LAN needs an explicit range and the same cleartext acknowledgement a node needs. Nothing is ever added without a person confirming it. |
| **P7** Background lifetime | **not started** | |

### What is NOT claimed

Stated here so nobody has to discover it by reading code:

1. **The §4 template divergence is not reproduced.** The original measurement (Camelid answers `12`
   where llama.cpp/Ollama/LM Studio answer `7`) used one pinned GGUF on every arm. The live receipt
   here has weights that are only *asserted* equivalent; both engines answered `12`, and the
   difference was in surrounding prose. **Reproducing it requires the same GGUF file loaded into
   both engines** — see §6, task R1.
2. **Mixed routing has no live receipt.** The policy is proven offline only.
3. **LM Studio's per-model `capabilities: ["tool_use"]` field is deliberately unused.** It appears
   in the real API response but not in the docs, and it is a vendor declaration about a model, not a
   measurement of the engine. Wiring it in would violate I5.
4. **One ablation (D5) is unguarded offline** and is declared so by the harness rather than counted.
   It is closed by a live receipt instead.
5. **An unauthenticated answer can be imitated.** "Answers like camelid" means one address produced
   this engine's health signature, nothing more. Anything on the network can produce that signature,
   and a NanoCamelid on this very LAN implements the same contract. The human confirm, the
   server-computed bearer warning and `identity_basis` on every finding all bound the consequences;
   none of them eliminates it. `--node-tls-ca` is the authenticated route, and under it a match
   records `certificate_verified`.
6. **Discovery's Windows paths have no live receipt.** Interface enumeration and reverse DNS are
   implemented for unix only: `local_interfaces` and `reverse_lookup` are `cfg(unix)`, and elsewhere
   they answer "not available on this platform", which surfaces as an unknown rather than a guess.
   The suggestion then falls back to a routed-address probe labelled `assumed_24`. The spec called
   for `GetAdaptersAddresses` and `ws2_32` bindings; those are not written, so no Windows-only FFI
   ships untested rather than shipping unverified.
7. **LM Studio's signature is proven offline only.** No LM Studio instance was reached in any live
   run, so its identification is exercised by fixtures taken from its documented API and by a stub.
8. **"Possibly the same machine" is never a claim.** Rows that answer identically across addresses
   are linked with `possibly_same_as` and the wording says this build cannot tell whether they are
   one machine. Nothing de-duplicates them, and nothing suppresses either row.

---

## 5. Testing standard — this is the part that matters

Passing tests are not the bar. The bar is **a test that would have failed if the code were wrong**.

### 5.1 Gates — all must be green before any push

```bash
cargo fmt --all -- --check                       # 0
cargo clippy --all-targets -- -D warnings        # 0
cargo test --lib fabric::                        # 458 passed, 2 failed (see below)
cargo test --test fabric_discover                # 19
cargo test --test fabric_serve                   # 94
cargo test --test fabric_end_to_end              # 18
cargo test --test fabric_engines                 # 39
cargo test --bin camelid                         # 71

cd frontend
npm run build
npm run smoke:fabric-model                       # 38 checks
npm run smoke:fabric-view                        # 44 checks
npm run smoke:divergence-model                   # 44 checks
npm run smoke:divergence-view                    # 40 checks
npm run smoke:discovery-model                    # 22 checks
npm run smoke:discovery-view                     # 14 checks
```

**Two `fabric::http` TLS tests fail on the M4 node and are not P4's.**
`an_untrusted_ca_and_a_wrong_server_name_are_refused_before_http` and
`bearer_bytes_are_not_sent_before_the_tls_peer_is_authenticated` both reach a
stub through the name `localhost`, which resolves dual-stack. `connect_tls_any`
keeps the *last* failure, so a dead `::1` sibling overwrites the certificate
error the test asserts on, and it reads as `Connect` rather than `Tls`. It is
unfixed on main and P4 touched none of that path. Everything else above is
green; `cargo clippy --all-targets -- -D warnings` is clean apart from one
pre-existing `unnecessary_cast` at `src/metal.rs:69514`, which appears under
Homebrew rustc 1.94.1 and not under the 1.95.0 CI pins.

**A count that goes down is a regression even if everything passes.** Record the new counts when you
add tests.

### 5.2 Ablations — mandatory for every honesty rule

Every rule that makes this feature trustworthy must be **deliberately broken** and caught by a
*named* check. A rule with no ablation is decoration.

Procedure:

1. Name, in advance, the check you expect to fail.
2. Break exactly one rule — one edit, one behaviour.
3. Run the gate. Confirm **that named check** fails, not merely that something failed.
4. Restore the file and **verify the restoration by SHA-256**.
5. If the predicted check passes, the rule is unguarded. Say so; do not quietly move on.

Rules already ablated and caught (8 ablations, 7 guarded, D5 declared unguarded):

| # | Sabotage | Caught by |
|---|---|---|
| D1 | call a single run "stable" | `one_run_a_side_is_unmeasured_rather_than_stable` |
| D2 | attribute a difference despite a self-contradicting side | `an_unstable_side_makes_a_difference_unattributable_and_suppresses_the_diff` |
| D3 | compare two different models as the same weights | `different_model_identities_are_reported_as_such_and_never_as_divergence` |
| D4 | claim LM Studio was seeded | `an_engine_without_a_seed_parameter_is_recorded_as_uncontrolled` |
| D5 | ask a node that does not hold the model | **unguarded offline — closed by live receipt** |
| D6 | accept a verdict kind the build does not recognise | `a verdict this build does not know is unknown, never read as agreement` |
| D7 | give an unstable side a settled digest | `an unstable side is not attributable and carries no settled digest` |
| D8 | render a proxy refusal as an empty comparison | `a refused comparison shows the refusal, not an empty result` |

P5 remainder (mixed-engine placement), run on the M4 node against `42561ea8`, one edit per row,
each restored and verified by SHA-256; every named check failed (receipt:
`receipts/p5-mixed/README.md`):

| # | Sabotage | Caught by |
|---|---|---|
| D9 | `meets()` passes every node whatever `tool_calls` says | `a_tool_calling_request_never_lands_on_a_backend_nobody_measured`, `a_request_carrying_tools_is_never_sent_to_an_unmeasured_engine`, `a_tool_calling_request_through_the_proxy_is_refused_with_what_to_do_next` |
| D10 | delete the requirements union in `Fabric::dispatch` (policy unit tests stay green) | `a_request_carrying_tools_is_never_sent_to_an_unmeasured_engine`, `a_tool_calling_request_through_the_proxy_is_refused_with_what_to_do_next` |
| D10b | the same, in `dispatch_streaming` only | `a_streaming_request_carrying_tools_is_never_sent_to_an_unmeasured_engine` |
| D29 | `forward_to` checks no requirement | `a_one_shot_send_refuses_a_tool_carrying_body_for_a_node_not_measured_for_it` |
| D11 | `placement_requirements` ignores `functions` | `contents_of_equal_encoded_size_cannot_change_requirements_or_class` |
| D12 | `placement_requirements` also scans messages for `role: "tool"` | `contents_of_equal_encoded_size_cannot_change_requirements_or_class` |
| D18 | drop the per-attempt `model` write | `a_forwarded_request_carries_the_node_s_own_model_id_and_nothing_else_changed` |
| D41 | the body rewrite also removes the client's `user` field | `a_forwarded_request_carries_the_node_s_own_model_id_and_nothing_else_changed` |
| D42 | `is_placeable_under` admits any node under Allowed, ready or not | `mixed_placement_refuses_an_unreachable_node_exactly_as_camelid_only_placement_does`, `mixed_placement_does_not_make_a_node_that_is_still_loading_eligible` |
| D15 | `--allow-mixed-engines` defaults to true | `fabric_serve_places_on_camelid_only_unless_the_flag_is_given` |
| D15b | `MixedEngines::default()` is `Allowed` | `the_proxy_places_on_camelid_only_unless_started_with_mixed_engines` |
| D33 | the flag reads `CAMELID_ALLOW_MIXED_ENGINES` | `fabric_serve_mixed_mode_cannot_be_turned_on_by_the_environment` |
| D13 | the re-place arm trusts `engine_queue_full` from any engine | `a_foreign_503_carrying_our_queue_full_code_is_still_relayed_once`, `a_streaming_request_refused_by_a_foreign_node_is_relayed_once` |
| D30 | dispatch's service-time branch uses the ungated predicate | `completion_time_invalidates_a_foreign_node_whose_503_only_looks_typed` |
| D43 | any 503 is read as a queue-full refusal | `a_refusal_that_is_not_backpressure_is_relayed_untouched`, `an_untyped_refusal_of_a_stream_is_relayed_once_not_re_placed` |
| D44 | a buffered answer to a stream is re-placed on any 503 | `an_untyped_refusal_of_a_stream_is_relayed_once_not_re_placed` |
| D45 | the mixed-mode model-less rule applied in every mode | `a_camelid_only_fabric_decides_exactly_as_before` |

Review follow-up: the spec rows the P5 pass left unrun, and the guards this review added (R1–R3).
Run on the M4 node against this branch's tree, one edit per row — except D14, whose behaviour needs
both the queue-full code check and the trust gate removed, so it carries two. Every named check
failed, every file was restored and verified by SHA-256 (9 files, all pristine at the end):

| # | Sabotage | Caught by |
|---|---|---|
| D14 | re-place any 5xx, from any engine (the queue-full code check and the `typed_backpressure` gate both removed) | `a_foreign_node_s_untyped_503_is_relayed_once_and_names_its_engine`, `a_refusal_that_is_not_backpressure_is_relayed_untouched` |
| D16 | `serves()` matches installed models without checking `loads_on_demand` | `an_installed_model_on_an_engine_whose_loading_is_unprobed_is_not_matched` |
| D17 | the probe reads a failed `/api/ps` as an empty list (`unwrap_or_default`) | `a_node_whose_running_models_cannot_be_read_still_serves` |
| D19 | `node_detail` `placeable` uses `is_placeable()` rather than `is_placeable_under(mixed)` | `health_under_mixed_mode_reports_the_mode_the_nodes_it_accepts_and_their_consequences` |
| D22 | count unadmitted ready nodes as unobserved again | `a_model_only_an_unplaced_engine_holds_is_refused_as_settled_and_names_it` |
| D23a | `forward_error` omits `x-camelid-fabric-engine` | `every_answer_through_a_mixed_proxy_names_its_engine`, `a_failure_through_a_camelid_only_proxy_carries_its_node_and_engine_and_nothing_else` |
| D23b | the buffered path looks the engine up in `fabric.specs()` by label after the failure | `a_failure_names_the_engine_that_was_sent_the_request_even_if_the_file_changed` |
| D24 | `COLD_LOAD_COST` set to 0 | `a_resident_holder_outranks_a_cold_one_at_equal_load` |
| D25 | `tag()` omits `x-camelid-fabric-model-identity` | `an_answer_resting_on_a_declared_alias_says_so_and_one_that_does_not_does_not` |
| D26 | `placement_blocker_keys()` reversed | `each_blocker_key_names_the_capability_that_produced_its_string` |
| D27 | `held_by_unplaced` dropped from the `ModelUnavailable` Display | `a_model_only_an_unplaced_engine_holds_is_refused_as_settled_and_names_it` |
| D31 | `route_error` uses Display regardless of `disclose` | `an_exposed_listener_refuses_without_naming_engine_versions` |
| D32 | `tag()` always emits `x-camelid-fabric-model` | `a_camelid_only_proxy_serves_a_tool_calling_request_exactly_as_before`, `a_camelid_only_answer_carries_no_new_header` |
| D34 | `x-camelid-fabric-residency-observed: resident` sent for `Unknown` | `residency_is_reported_as_observed_and_absent_when_unknown` |
| D35 | model-less requests admit every ready node under mixed mode | `a_request_naming_no_model_is_placed_only_where_the_model_it_will_get_is_known` |
| D36 | `server.rs` `model()` calls the free `route()`, which never sees the aliases | `an_aliased_model_is_retrievable_and_listed_exactly_when_it_would_be_placed` |
| D37 | the `foreign_additions` call removed from the node-set reload | `a_foreign_node_added_under_mixed_mode_is_announced`, `a_foreign_node_added_while_mixed_is_announced_in_health` |
| D38 | the Camelid `rerank_route` detail says rerank is supported | `a_route_existing_is_not_the_model_supporting_it` |
| D38b | Ollama credited with a rerank route | `a_rerank_request_never_lands_on_an_engine_without_the_route` |
| R1 | `shared_weights_digest` lets an unconfirmed binding vouch again | `a_served_binding_vouches_for_nothing_when_its_side_published_no_digest`, `a_node_that_served_a_binding_but_published_no_digest_is_not_verified_by_it` |
| R2 | `engine_and_version` says "version not published" for every node without a version | `a_node_that_was_not_reached_has_an_unknown_version_not_an_unpublished_one`, `health_calls_an_unreached_nodes_version_unknown_never_unpublished` |
| R3 | the unplaced holders are not filtered by what the request needs | `a_holder_the_flag_would_still_refuse_is_never_offered_the_flag` |

P4 discovery. Twenty rows were run on the M4 node, one edit each unless noted,
every file restored and verified by SHA-256, with a preflight before each row
confirming the tree was pristine. A row counts as caught only when **that named
check** failed and the log carried `test result:`, so a gate that silently did
not run cannot pass for a guard:

| # | Sabotage | Caught by |
|---|---|---|
| X1 | the default scope also covers a LAN /24 | `the_default_scope_is_loopback_only` |
| X2 | remove the private-range allowlist | `a_public_range_is_refused_before_any_socket` |
| X3 | accept an oversized range instead of refusing it | `a_range_over_the_address_limit_is_refused_not_truncated` |
| X4 | `plan()` skips the node-transport preflight | `cleartext_to_the_lan_is_refused_without_the_acknowledgement` |
| X5 | the scan presents the fabric bearer to whatever answered | `no_host_receives_a_credential_it_was_not_declared_to_receive` |
| X6 | the CLI reads `CAMELID_API_KEY` and sends it | `the_cli_never_reads_camelid_api_key_for_discovery` |
| X10 | give an unmatched HTTP service an engine proposal | `an_html_page_is_other_http_with_no_proposal` |
| X13 | skip the `base_sha256` comparison | `a_file_changed_since_the_scan_is_refused_and_untouched` |
| X14 | re-serialize the file from parsed specs | `joining_appends_and_leaves_every_existing_byte_alone` |
| X18 | remove the loopback-peer guard | `a_remote_peer_cannot_trigger_a_scan` |
| X19 | remove the Host-header guard | `a_rebound_host_header_is_refused` |
| X20 | remove the Origin guard and rely on CORS | `an_origin_not_on_the_cors_list_cannot_trigger_a_scan_or_a_join` |
| X21 | the join trusts the client's classification | `a_join_refuses_an_engine_the_host_no_longer_answers_like` |
| X22 | a disabled proxy stops being tellable from an old build | `discovery_is_off_unless_asked_for` |
| X34 | remove the duplicate-endpoint check | `a_second_label_for_an_existing_endpoint_is_refused` |
| X39 | the join's re-identification presents the bearer | `no_host_receives_a_credential_it_was_not_declared_to_receive` |
| X46 | remove the `answered_from` check at join | `a_join_refuses_a_name_that_reaches_another_address` |
| X50 | a non-TTY run prompts and defaults to yes (two edits: the TTY gate and the default) | `a_non_tty_run_never_writes_without_join` |

**X30 is the interesting one, and it is reported as it happened.** Removing
only the post-append spec-list check — the sabotage as specified — did *not*
fail its named test: those fixtures are refused earlier, by the comment
sanitiser. Rather than record a guard that is not doing the work, two further
rows establish what each layer is worth:

| # | Sabotage | Result |
|---|---|---|
| X30a | remove the comment sanitiser, keep the post-append check | test still passes — the backstop alone refuses it |
| X30b | remove both | `an_append_that_would_add_anything_but_the_requested_node_is_refused` fails |

So the nodes file has two independent layers in front of it and either one is
sufficient. That is a stronger result than the single row would have given, and
it is the reason the single row looked unguarded.

**X12 has no offline guard and is not claimed to have one.** The rule is that a
reverse-DNS name is listed only after a forward lookup confirms it. Proving it
offline needs an injectable resolver, which this build does not have:
`prove_name` calls `netscope::forward_lookup` directly. It is closed by the
live receipt instead, and the seam is worth adding before anyone relies on the
rule in a unit test.

**Any new guard in P4/P5/P7 needs the same treatment.**

### 5.3 Receipts — against real software, not stubs

A stub proves the code does what you told it to. A receipt proves the *world* behaves as you
assumed. Every phase that touches a real engine needs one.

**Always take a negative control first.** The strongest receipt in this work is one Ollama server
under two labels answering `IDENTICAL` — if the tool reported a difference between a server and
itself, every other verdict it produced would be worthless.

### 5.4 Regression tests are as important as guards

Every guard needs a paired "existing behaviour is unaffected" test. A safety feature that quietly
changes what already worked is its own bug. This caught a real one: gating tool calls on
`provenance == Measured` would have disabled tool calling on **every Camelid build** except the
single version in the measurement table.

---

## 6. Work remaining, in priority order

### R1 — Reproduce the §4 template divergence *(small, high value)*

The one open gap inside what is already shipped.

**Do:** load the *same GGUF file* into both engines, so the weights are identical rather than
asserted. LM Studio stores models under `~/.lmstudio/models/<publisher>/<repo>/*.gguf`; Ollama can
build a model from that exact path with a `Modelfile` containing `FROM /path/to/file.gguf`. Verify
with a sha256 of the file, then run `fabric compare` at temperature 0.

**Exit:** a receipt showing different sha256 answers from the same file, with both templates
captured and the differing line visible. If it does **not** reproduce, say so and record what the
answers actually were — that is a finding either way.

### R2 — P5 remainder

**Do:** GUI Screen E (routing mode selector, defaulting to Camelid-only, with a confirmation that
states in product language what mixed mode accepts — derived from the live capability matrix, not a
static paragraph that can go stale). Then a live receipt of mixed placement across two real
backends. Also still missing: **relay-don't-re-place** — an untyped refusal from a foreign node must
be relayed once, not retried on a sibling, because it cannot be distinguished from a real failure.

**Exit:** ablation proves each guard fires; live receipt on two real backends; a tool-calling request
demonstrably never lands on an unmeasured backend.

### R3 — P4 discovery *(done — receipt below)*

`fabric discover` and GUI Screen B share one implementation and one serialized `Discovery` value,
so the terminal and the browser cannot describe different networks. Confirm-before-join throughout.

**Live receipt.** Run between two M4 machines on a lab LAN. Every target sat behind an ~80-line
stdlib Python recorder that logs each `accept()` and every client-to-server byte, because the claim
being tested is about what is *not* on the wire and neither Ollama nor `http.server` records request
headers. The engines stayed bound to loopback; the recorder is what the scanner talked to. The
binary was built on the second machine and copied, matching sha256 on both
(`fa139257bb86934e…`); nothing was built on the machine the scan ran from.

| Claim | How it was shown |
|---|---|
| The three kinds are classified correctly | Over a 14-address slice × 4 named ports: `answers_like camelid 0.7.3`, `answers_like ollama 0.33.2`, `other_http` for the unrelated service, `silent_after_connect` for a port that accepts and never writes. The unrelated service is never called an unknown engine. |
| A fourth kind, unprompted | A second address on the same slice answered identically and was linked with `possibly_same_as`, with the wording that this build cannot tell whether they are one machine. Neither row was merged or suppressed. |
| Nothing is claimed that was not reached | `planned=64`, `8 findings + 12 refused + 44 timed out + 0 not_scanned = 64`. The accounting closes exactly. |
| The default scope touches nothing off-box | Recorder accept counts captured before and after a zero-argument run were byte-identical. The run reported 2 addresses × 3 engine ports, all refused locally. This is accept-level evidence, not an application log. |
| A LAN needs the acknowledgement | `--cidr <lab range>` without it: exit 2, naming both `--allow-cleartext-node-transport` and `--node-tls-ca`, before any packet. |
| Public ranges are refused outright | `--cidr 8.8.8.0/24` *with* the acknowledgement: exit 2, listing the ranges this build will scan. |
| No credential reaches a scanned host | `CAMELID_API_KEY` was set to a marker for every scan. Across all four recorders: marker `0`, `authorization` `0`, and — the positive evidence that the scan actually arrived — `User-Agent: camelid-fabric-discover/` `16`. An earlier run of this same check read "clean" only because the scan had not run; the UA column is what caught it. |
| The bearer warning is true in both directions | One probe round over a fabric holding both engines: the Camelid node's recorder shows the marker and an `Authorization` header (`bearer_will_be_sent`); the Ollama node's shows neither. |
| Looking adds nothing | The nodes file's sha256 was identical before and after two full scans. |
| The nodes file is the only thing written | A join left the first 44 bytes byte-identical and appended exactly a comment line and a node line; the directory listing was unchanged, so no temp, lock or backup file survived. `fabric status` then probed the joined node `ready`, so the loader accepts what discovery wrote. |
| Refusals leave the file alone | `duplicate_endpoint` (naming the existing label and pointing at hand-editing), `no_longer_answers` (the HTTP service claimed as an engine), `invalid_label` (`#x`), `invalid_host` (a host with a space). |
| Listed names are proven from the scanning host | The one reverse-DNS name found was forward-resolved from the scanning machine and the answering address was in the set (membership, not order). Its proposal still used the **address** as the host, with the name offered only as an alternative carrying the "your router chose this" warning. |
| It agrees with itself | Two consecutive scans produced identical classifications for every address and port. |

**Not established, and not claimed.** No LM Studio instance was reached, so that signature is
offline-only. The GUI was exercised by its browser smoke against a scripted proxy rather than by a
hand walkthrough against a live one. The Raspberry Pi on this network was deliberately left out of
the scanned range. And X12 has no offline guard — see §5.2.

**Care:** this is the highest-risk surface in the lane — it sends traffic to machines the user did
not name. Never present the fabric's bearer token to an unidentified host.

### R4 — P7 background lifetime

**Do:** the desktop app currently kills the sidecar on window close (plus a Windows job object). A
cluster host that dies when its window closes cannot serve other devices — which contradicts the
premise of the whole feature. Give it tray or background lifetime with an explicit quit.

**Exit:** closing the window leaves the engine serving; quitting stops it; the tray states which it
is.

**Care:** verification is the hard part, not the code. Do not ship this on "it compiles".

### R5 — Measurement mode *(the strategic one)*

P3 built a capability matrix that says `not_probed` for every foreign engine. P6 built the machinery
that can probe. Close the loop: let a comparison **earn** a `measured` provenance and record it
against that exact engine version. Nobody else does this, and it turns "we refuse to claim what we
have not measured" from a limitation into the feature.

---

## 7. Traps — apparatus failures that produce confident, wrong, green results

Every one of these actually happened during this work. Each produced a result that *looked* fine.

1. **`$args` is a PowerShell automatic variable.** `function G($name,$args)` meant cargo received no
   arguments, printed help, and exited 0 — seven gates "passed" without running. **A gate that
   succeeds without emitting its characteristic output (`test result:`) has not run.**
2. **`execFileSync` defaults to a 1 MiB output buffer.** A full cargo rebuild overflows it, the call
   throws, output is truncated, and "no FAILED lines" reads as "the rule is unguarded". Set
   `maxBuffer` and require positive evidence the tests ran.
3. **A killed or overlapped ablation run leaves the tree sabotaged.** The SHA-256 restore check only
   protects a run that *finishes*. Always preflight for leftover sabotage before measuring anything.
   **Never run two harnesses concurrently.**
4. **This clone checks out CRLF.** A multi-line search anchor written with LF silently never matches,
   and a sabotage that never happened looks exactly like one that was caught.
5. **A narrow test run can pass against a stale build.** `cargo test --bin camelid` reported 61/0
   while the binary genuinely could not compile (missing re-exports). Only a full-suite clean build
   exposed it.
6. **Log filters eat real output.** A receipt that stripped `^\s*\+ ` to remove PowerShell error
   markers also ate every added line of a diff. Write child stdout straight to a file.
7. **`Start-Process -ArgumentList` joins on spaces without quoting**, so a prompt containing spaces
   arrives as several arguments.
8. **Do not change `core.autocrlf`** in this clone. A QA script hashes the working tree, so flipping
   it breaks a pinned digest with zero source changes.

---

## 8. Environment facts, verified on the development machine

| | |
|---|---|
| Ollama | port **11434**; `GET /api/version`, `/api/tags`, `/api/ps`; `POST /api/chat` with `options.{temperature,seed,num_predict}`; `POST /api/show` returns the model's **`template`**. No queue depth anywhere in the API. |
| LM Studio | port **1234**; `GET /api/v0/models` (needs ≥ 0.3.6) with per-model `state` = `loaded` \| `not-loaded`; `POST /api/v0/chat/completions` takes `temperature`/`max_tokens` and returns a `runtime` block. **No version endpoint. No seed parameter. No prompt-template endpoint.** |
| Camelid | `GET /props` → `chat_template`; `POST /apply-template` renders messages without inference; `ChatCompletionRequest` accepts `temperature`, `top_p`, `seed`, `max_tokens`. |
| Model identity | Ollama's `/api/tags` `digest` is a **manifest** digest, not the GGUF's. LM Studio publishes none. **Digest matching across engines is not available** — this is why O6 is operator-declared. |

---

## 9. House style for this lane

- Comments say **why**, never what. If the code shows it, do not write it.
- Prefer deleting a dishonest feature to fixing it. P1 removed more than it added.
- A refusal must name what to do next. `"studio does not hold X; it holds A, B, C"` beats
  `"model not found"`.
- Never let a failure read as an empty success. `404` on a route the proxy does not have is not the
  same as "no results".
- When collapsing test literals into a helper, check the literal was not itself the point of the
  test.
