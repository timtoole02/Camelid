# Gemma4 post-reboot checkpoint through O3 — 2026-08-24

Machine: 16 GiB Apple M4. All throughput below is decode tok/s on the frozen
48-token fixture. **Exact** means all 48 generated token IDs match. **Ceiling
only** means target-verified output proposed with oracle fixture knowledge; it
is deliberately non-production and cannot be promoted as model throughput.

## Mini2 result after this checkpoint

On the clean second M4, H11 settled at a 25.08 tok/s four-run mean before the
MTP correction. Fixing the assistant's proportional full-RoPE geometry reduced
the frozen exact request from eight verifier rounds to seven: corrected runs
were **26.85 and 27.66 tok/s** around a preserved-old-binary control at
**24.83 tok/s**. This is a +9.8% paired production gain, not an oracle result.

The clean Mini2 O3 ceiling reached **34.31 and 35.26 tok/s**, exact but still
fixture-oracle/non-promotable. The remaining production miss is the first
answer-format token: the assistant strongly selects unfenced `pub`; the target
selects a Markdown code fence. The target token is assistant rank 2 but is
6.405 logits behind, ruling out a tiny calibration tweak. See the current
addendum at the top of `HANDOFF-2026-08-24.md` and the `mini2-h*-ropefix-*`
receipts for the newer state.

## Results

| experiment | correctness / status | decode tok/s | conclusion |
|---|---|---:|---|
| H5, 2,100-record capacity, three runs | **exact 48/48** | **22.77, 22.58, 22.48** | Stable capacity point; keep as the conservative enlarged-hot reference. |
| H4, 2,200-record capacity | **exact 48/48** | peak **23.88** | The peak did not repeat (22.35, then 21.64). H4 also reserves 100 more expert records than H5 and finished at only 26–27% free memory, so it is not a safe default from this evidence. |
| H2 embedding predictor probe | **exact 48/48, observation only; throughput contaminated** | not promotable | **NO-GO.** At the real global cap, cap-64 recalled 5.84% of residual misses at 8.07% precision; cap-96 recalled 7.16% at 6.60% precision, far below the 30% implementation floor. Do not build embedding-driven host staging. |
| Historical O1 K8 / O2 K16 | **exact 48/48, ceiling only, non-production** | **22.33 / 23.58** | Historical oracle-seeded ceilings only. These receipts used the old loader that stripped the seed's final LF; they are not byte-identical evidence for the hardened loader. |
| H8 overlap-on / H2 control / H8 overlap-on | **exact 48/48 / exact 48/48 / exact 48/48** | **22.12 / 20.29 / 22.74** | Tight H2 sandwich: overlap averaged 22.43, **+10.5%** over the control. This is the strongest same-state evidence for the hot/cold overlap mechanism. |
| H9 overlap with `MIN_COLD=2` | **non-exact; prefix only 4/48** | invalid | Correctness reject. Its observed 21.01 is not a throughput result. |
| H11 no-publish / publish-on anchor / H11 no-publish | **exact 48/48 / exact 48/48 / exact 48/48** | **23.15 / 22.91 / 23.90** | No-publish is promising, but the small sandwich delta remains inside known machine-state variance; reproduce after reboot before promotion. |
| H12 2,100 no-publish | **exact 48/48** | 20.98 | No promotion: the isolated run was slower and extra capacity raised memory pressure. |
| H13 1,546 no-publish | **exact 48/48** | 19.23 | No promotion: cutting residency raised misses from 146.0 to 193.2 per round. |
| H14 H2 no-publish, 12 read threads | **exact 48/48** | 19.90 | No promotion: read throughput was unchanged at 4.26 GB/s, so extra workers exposed no I/O gain; the isolated tok/s delta is machine-state contaminated. |
| O3 H2 overlap, no-publish, K8 oracle | **exact 48/48, ceiling only, non-production** | **25.22** | Useful ceiling observation, not an absolute settled ceiling: the machine was already swap-contaminated (661 MiB used) and the target GPU term had slowed to about 200 ms/round. |

## Machine-state boundary

Absolute numbers moved materially with unchanged profiles: H4 ranged from
23.88 down to 21.64, and later runs showed slower GPU terms despite unchanged
read geometry. Relative, tightly interleaved sandwiches are stronger evidence
than isolated peaks. A fresh reboot is still required before publishing a final
absolute number: warm the lane, then interleave exact H11 no-publish and H8
publish-on controls, followed by a fresh O3 ceiling receipt.

## Corrected prefill diagnosis

The earlier “about 85 records per layer, therefore prefill binds mapped
records” explanation is stale. At K8 a layer's exact routed union is bounded at
64; H2 demand-promotes that exact union, and mapped readahead is zero. Also,
`load_s` is server startup/readiness time, not part of the per-request
`http_wall_s`, so it must not be counted as request prefill.

Prefill is still a large end-to-end opportunity, but the remaining mechanism is
cross-chunk churn: the 104-token prompt spans 13 K8 chunks and H2 performs 2,294
record reads / 7.32 GiB, with roughly 3.96 s of honest `slot_filler` wall time.
Work on retention/admission across chunks or chunk geometry, not a nonexistent
over-64 mapped-union bind.

## Next levers

1. Reboot and reproduce the exact H11/H8/H11 sandwich and O3 before deciding
   whether no-publish becomes the new overlap default.
2. Stop the failed embedding predictor path. If prediction is revisited, first
   measure a stronger future-route signal observation-only; do not mutate or
   publish the directory from a prediction.
3. Attack both sides of the 35 tok/s gap. O3 shows that perfect K8 proposals
   alone are insufficient in the measured state, while the learned K9–K16
   width sweep already lost to K8. A credible path needs better useful alpha
   without proportional union growth plus lower target GPU cost, especially
   gate-up/QKV work.
4. For end-to-end latency, pursue prefill retention/admission or chunk geometry
   separately from decode tok/s. Do not use startup `load_s` or per-thread-sum
   `disk_time` as the optimization target; use request wall and `slot_filler`.
