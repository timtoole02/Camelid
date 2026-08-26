# H69 final result — 2026-08-25

## Outcome

The campaign stopped without reaching 50 tok/s. The highest exact throughput
observed anywhere in this Mini2 effort remains **36.84 tok/s** from H49's
2,100-slot live-hidden sequential staging run. That run produced the frozen
48/48 token sequence and observed zero swap, but it is a historical peak rather
than a promotion result: the later physical-footprint treaty rejects the
2,100-slot lane, and the repeatable historical H40 checkpoint was 35.02 tok/s
mean with a 35.28 tok/s peak. See
[`mini2-50tps-checkpoint-2026-08-24.md`](mini2-50tps-checkpoint-2026-08-24.md)
for the original receipts and decomposition.

The best observation under the literal 1,408-slot profile was an ad hoc
32.2565 tok/s selector stack with no dedicated checked-in profile. It combined
the request-local H71 prompt-ranked handoff with the retained H58 GateUp MMA,
H60 BF16 producer fusion, and H62 BF16 lattice-load selectors. It was exact
48/48 at 1,488.07 ms, but it was a single manual no-watchdog observation and is
not proposed for promotion.

## Literal 1,408-slot strict-profile observations

These observations were manual and non-qualifying; “strict-profile” describes
the configured physical footprint, not continuous safety qualification.

Every throughput below is `48 / sum(four MTP round walls)`. Every counted run
produced the exact expected 48-token sequence with the required effective
schedule `14,13,14,7`.

| experiment | request wall | tok/s | result |
|---|---:|---:|---|
| H49 clean control for H70/H71 | 1,562.20 ms | 30.7259 | reference |
| H71 prompt-ranked handoff | 1,514.12 ms | 31.7016 | −48.08 ms; single observation |
| H71 + retained H58/H60/H62 selectors | 1,488.07 ms | **32.2565** | exploratory strict-profile peak |

The H69 cap-8/cap-16/cap-16/cap-8 counterbalanced run measured:

| lane | observations | mean tok/s | delta |
|---|---|---:|---:|
| cap 8 | 30.909512, 31.123359 | 31.016436 | control |
| cap 16 | 30.997339, 31.289315 | 31.143327 | +0.409112% |

Cap 16 was rejected: its measured gain was too small to promote.

H70 paired an exact residency trace with the clean H49 control. It observed
661.535 ms of wave-load wall. Even a fixture-specialized, full-future-route,
oracle-weighted fixed 1,408-slot profile produced a linear route-coverage
projection of 1,100.975 ms, or **43.5977 tok/s**. This is not a physical
throughput ceiling: the trace records route-minus-residency identities while
the priced wave wall is measured after staged-ready substitution, so it lacks
the identity binding needed to price direct demand exactly. The analyzer marks
the mismatch as `physical_demand_identity_bound=false` and the result as
`throughput_promotion_allowed=false`.

H72 traced the real H71 handoff. Its four wave-load totals were 216.918,
173.254, 144.163, and 59.599 ms. Round zero alone exceeded the approximately
59 ms total I/O margin available at 50 tok/s. The current H71 implementation
therefore did not demonstrate a standalone path to the target.

## Safety and provenance

At the user's direction, these final observations did not run a watchdog.
`run_cfg.zsh` instead captured pre-launch, readiness, and post-request point
samples. The receipts are deliberately marked `qualifying=false` and
`point_samples_only=true`; they cannot prove what happened between samples.
All recorded samples passed the configured pressure, footprint, wired-memory,
swap-counter, process-group, and report checks.

The H70–H72 measurements used:

- source commit: `1dc1263a76f45cb2dc803085cd91b2fd04016234`
- binary SHA-256: `3c68e97dad708006e8cdaff7eda56186113fa0df2fea7c6724c929b8d03be5a4`
- binary version: `camelid v0.6.1-334-g1dc1263a7`
- binary size: 29,012,000 bytes

Raw run directories remain private on Mini2 because their manifests and logs
contain local paths and process metadata. This document contains only the
sanitized aggregate results needed for review.

## Verdict

- Historical exact peak: **36.84 tok/s**, footprint-inadmissible under the
  later treaty.
- Best 1,408-slot strict-profile manual observation: **32.2565 tok/s**, ad hoc
  and non-qualifying.
- 50 tok/s: **not achieved**.
- H69 cap 16: no-go.
- Fixed prompt/future-route residency: no promotion justified by the current
  evidence.
- All new runtime behavior remains default-off and exact-router-authoritative.
