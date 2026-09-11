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
sed -n '1,/^#\[cfg(test)\]/p' src/fabric/policy.rs | grep -nE 'Ollama|LmStudio'
```

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
| `d7fc6278` | P7 review — an engine seen to stop stays stopped in the tray; generations open through the host |
| `16c930db` | P7 review — single-instance plugin on Windows only (its macOS socket is a shared `/tmp` path) |
| `8737c504` | P7 — Windows lifetime smoke in CI with a stand-in engine (not yet run on a runner) |
| `1c4445d7` | P7 — opt-in background lifetime: tray, observed status, reapable pending sidecar, single instance |
| `35e0612b` | P7 — `camelid serve --exit-when-stdin-closes` (opt-in, no env var) |
| `68583ae3` | P7 — closing the main window quits again (the v0.7.0 hidden-Spotlight regression) |
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
| **P4** Discovery | **not started** | |
| **P7** Background lifetime | **code done; live receipts owed** | Closing the window quits by default again. That reverts a v0.7.0 regression: the hidden Spotlight window kept 0.7.x alive after a close (live receipt on 0.7.3 in the desktop README). The tray's opt-in "Keep engine running when window closes" hides the window and keeps the engine. Status is observed, never assumed. A starting sidecar is reapable, and `serve --exit-when-stdin-closes` is the macOS crash backstop. **The macOS GUI receipt for the new build and the real-engine Windows receipt are not taken yet, and the Windows code has not yet compiled or run on a runner.** |

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
5. **P7's live receipts are still owed, with one exception.** The background close has now been
   driven twice on macOS 26 with the bundle built from this branch: it hid the window and left the
   engine serving, correctly, but no notice was displayed at any point (CoreGraphics window list:
   every app window `onscreen=0` for 6 s after the close, and none after reopening) while
   `desktop-lifetime.json` recorded `notice_shown: true`. That is the defect A26/A27 now guard. The
   fix itself is unobserved, and the rest of the macOS GUI walk-through (close quits, reopen, Quit,
   crash) is a checklist for the next session, not a result.
6. **P7 on Windows is unverified at every level.** The `cfg(windows)` code has been reviewed and
   rustfmt'ed but never compiled: the job object's error handling, `launch_contained`'s
   containment, the single-instance registration, and the job test. The CI smoke that drives the
   real desktop with a stand-in engine has never run on a GitHub runner. The unit tests have passed
   on macOS only. A receipt from the PR author's Windows machine with the real engine is required
   before P7 is called done.
7. **"Loopback only" is the literal bind, not an access boundary.** A page in a local browser can
   reach a loopback port by DNS rebinding, and the engine's generation and health routes do not
   check `Host`. Background mode makes that exposure last as long as the app runs. No Host check
   was added (declined for P7).
8. **The macOS crash backstop exists only when the bundled engine advertises
   `--exit-when-stdin-closes`.** An older engine starts without it; a killed desktop then leaves
   that engine running.
9. **App Nap's effect on the sidecar itself is unmeasured.** The desktop holds an activity while
   backgrounded; whether the child engine is throttled is not established. Removing the activity
   is unguarded offline (A20 below).
10. **No serving of other devices.** The desktop never binds a non-loopback address.
11. **No single instance on macOS outside LaunchServices.** `tauri-plugin-single-instance` puts its
    macOS socket at a fixed path in the shared `/tmp`, where another account's socket can make a
    launch skip the check silently or swallow every launch. So it is registered on Windows only.
    Dock, Finder and `open -a` relaunches reach the running app as Reopen. A direct exec,
    `open -n` or a second copy of the app starts a second desktop and engine.

---

## 5. Testing standard — this is the part that matters

Passing tests are not the bar. The bar is **a test that would have failed if the code were wrong**.

### 5.1 Gates — all must be green before any push

```bash
cargo fmt --all -- --check                       # 0
cargo clippy --all-targets -- -D warnings        # 0
cargo test --lib fabric::                        # 322 passed, 0 failed
cargo test --test fabric_serve                   # 74
cargo test --test fabric_end_to_end              # 17
cargo test --test fabric_engines                 # 15
cargo test --bin camelid                         # 67 (includes serve_optional_desktop_flags_default_off)
cargo test --test serve_stdin_close              # 2
cargo clippy -p camelid-desktop --all-targets -- -D warnings   # 0
cargo test -p camelid-desktop --all-targets      # macOS: 48 unit + 10 installer_hooks + 14 lifetime_guards
                                                 # Windows: NOT YET RUN on any runner; expected
                                                 # 51 unit (+ verbatim-path x2, job object) + 10 + 14

cd frontend
npm run build
npm run smoke:fabric-model                       # 27 checks
npm run smoke:fabric-view                        # 26 checks
npm run smoke:divergence-model                   # 18 checks
npm run smoke:divergence-view                    # 18 checks
```

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

**Any new guard in P4/P5/P7 needs the same treatment.**

P7: 33 ablation rows run at `d7fc6278` on mini2, all caught (the review round; the first 27
ran at `8737c504`). Each made one behavioural edit (one file, or the manifest plus its
registration for A24) to a synced copy, never the tracked checkout, and ran one gate. In 32
rows the named test failed; A5m failed to compile with E0624 (`begin_epoch` is private), which
is its guard. For the other rows the "compile_errors=1" the harness prints is cargo's
`error: test failed, to rerun …` summary line, not a compile error. Every file was restored and
verified by SHA-256 after each row and again at the end. Rows marked *source* are text guards
in `tests/lifetime_guards.rs`. They keep the shape from regressing quietly, but only a live
receipt proves the behaviour. In the first round A5b and A14 sabotaged helpers, not the
production sites the spec names, so deleting the real call left every test green. This round
ablates the production sites (A5b, A5s, A5m, A14) and keeps the helper rows as A5h and A14h.
A10 and A10b (`serve --exit-when-stdin-closes`) were not re-run: `src/main.rs` is unchanged
since they were caught at `8737c504`.

A26, A26p and A27 are new, and ran on the background-notice fix at the head of this branch.
They guard the one thing the first macOS GUI walk-through actually caught: the notice was
raised *after* the window hid, and unparented, so nothing was displayed while the preference
recorded that it had been (see "What is NOT claimed" 5).

| # | Sabotage | Caught by |
|---|---|---|
| P7-A1 | preference defaults to keeping the engine | `lifetime_preference_defaults_to_closing_the_engine_with_the_window` |
| P7-A2 | `close_action` hides with background mode off | `close_with_background_off_quits_the_app_rather_than_destroying_the_window` |
| P7-A2m | main.rs lets an OFF close proceed instead of exiting | `closing_with_background_off_requests_the_exit` (*source*) |
| P7-A3 | skip the stderr drain | `a_chatty_sidecar_never_blocks_on_a_full_stderr_pipe` |
| P7-A4 | a live process alone reads Running | `running_requires_a_fresh_health_answer` |
| P7-A4b | a stale answer reads not answering | `a_stale_observation_without_a_failure_reads_checking` |
| P7-A4c | a recent answer outranks an observed exit | `a_sidecar_that_exited_is_reported_stopped_not_running` |
| P7-A4d | failed probes never count | `running_requires_a_fresh_health_answer` |
| P7-A5 | the status store ignores the epoch | `a_health_result_from_a_replaced_engine_is_discarded` |
| P7-A5b | `EngineHost::begin_restart`, the only restart path, skips publishing Restarting | `a_restart_moves_the_tray_to_the_new_engine_and_reaps_the_old_one` |
| P7-A5s | `EngineHost::begin_start` opens the first generation without moving the tray's store | `a_restart_moves_the_tray_to_the_new_engine_and_reaps_the_old_one` |
| P7-A5m | `restart_engine` takes an epoch without `begin_restart` | does not compile: `begin_epoch` is private |
| P7-A5h | the `restart_transition` helper keeps the old status | `a_restart_never_publishes_the_previous_port` |
| P7-A6 | an unparseable body renders "No model ready" | `a_busy_health_body_never_claims_no_model_is_loaded` |
| P7-A6b | the busy body renders "No model loaded" | `a_busy_health_body_never_claims_no_model_is_loaded` |
| P7-A7 | drop `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` | `sidecar_is_never_detached_from_its_job` (*source*); the behavioural `job_object_kills_the_sidecar_when_its_last_handle_closes` needs a throwaway windows-latest run, **not done** |
| P7-A8 | `api.prevent_exit()` on ExitRequested | `desktop_never_prevents_app_exit` (*source*) |
| P7-A9 | remove `.show_menu_on_left_click(false)` | `tray_menu_does_not_steal_the_spotlight_click` (*source*) |
| P7-A10 | start the stdin watcher without the flag | `serve_without_the_flag_survives_a_closed_stdin` (at `8737c504`; not re-run) |
| P7-A10b | never start the stdin watcher | `serve_exits_when_its_stdin_closes_under_the_flag` (at `8737c504`; not re-run) |
| P7-A12 | Windows background without the job | `windows_background_requires_a_kill_on_close_job` |
| P7-A13 | background without a tray | `background_requires_a_live_tray` |
| P7-A14 | `start_with`, the body of `EngineHost::start`, holds the child outside the slot until its gate passes (the pre-P7 shape) | `shutdown_during_the_health_gate_reaps_the_child`, driven through `start_with` |
| P7-A14h | the `reap` helper skips a pending sidecar | `shutdown_during_the_health_gate_reaps_the_child` |
| P7-AQ | quit reaps nothing | `quit_reaps_a_running_engine_and_refuses_new_starts` |
| P7-AQm | the tray's Quit does nothing | `every_quit_path_reaches_the_engine_shutdown` (*source*) |
| P7-A15 | the flag changes before the write succeeds | `a_failed_preference_write_leaves_the_effective_setting_unchanged` |
| P7-A16 | pass `--exit-when-stdin-closes` unconditionally | `optional_flags_are_passed_only_when_advertised` |
| P7-A19 | `get-desktop-macos.sh` hard-kills the app | `upgrade_scripts_still_quit_the_app_and_wait_on_the_sidecar` |
| P7-A21 | the single-instance callback reads argv | `single_instance_callback_ignores_foreign_arguments` (*source*) |
| P7-AR | drop the macOS Reopen arm | `reopen_and_a_second_launch_show_the_main_window` (*source*) |
| P7-A22 | `StatusStore::apply` lets a live status replace Stopped in the same generation | `an_observed_exit_is_never_replaced_by_a_stale_live_snapshot` |
| P7-A23 | `EngineHost::supervisor_tick` returns the snapshot taken before its probe | `a_supervisor_tick_reports_an_exit_that_happened_during_its_probe` |
| P7-A24 | compile and register the single-instance plugin on macOS (manifest section and `#[cfg(windows)]`) | `single_instance_plugin_is_windows_only` (*source*) |
| P7-A25 | the tray's Restart item reroutes to the lifetime toggle | `every_tray_menu_action_reaches_its_handler` (*source*) |
| P7-A26 | hide the window before the notice is raised | `the_background_notice_is_raised_on_the_visible_window_before_it_hides` (*source*) |
| P7-A26p | raise the notice with no parent window | `the_background_notice_is_raised_on_the_visible_window_before_it_hides` (*source*) |
| P7-A27 | record `notice_shown` before the dialog is answered | `notice_shown_is_recorded_only_from_the_notice_callback` (*source*) |

Declared **unguarded offline**, each closed only by a receipt that is not taken yet:

- **P7-A11**, removing the single-instance plugin on Windows: the text guards see only the
  registration and its callback. It is closed by the CI smoke's `SINGLE-INSTANCE` check, which
  has not run yet. On macOS the plugin is not registered at all (A24).
- **P7-A20**, removing the App Nap activity: closed by an idle receipt on macOS (Activity
  Monitor's App Nap column, and the tray status after the idle period).
- P7-A17 and P7-A18 do not apply: they guarded the declined Host check.

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

### R3 — P4 discovery

**Do:** `fabric discover` plus GUI Screen B sharing one implementation. Confirm-before-join
throughout (I8).

**Exit:** scanning a LAN containing a Camelid node, an Ollama node and an unrelated HTTP service
classifies all three correctly; nothing joins without a click; the nodes file is the only thing
written; listed names are proven resolvable from the scanning host.

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
