# Opt-in CUDA min-p sampling

`CAMELID_CUDA_MIN_P_SAMPLING=1` enables a resident CUDA sampling tail for
`0 < min_p <= 1` with the default candidate floor (`min_keep` absent or one).
It requires the existing CUDA resident decode and temperature sampling
lanes. The new tail is disabled by default pending NVIDIA device qualification.
macOS continues to use its CPU sampler for min-p and its existing Metal paths.

The tail filters the untempered logits, then applies temperature and a seeded
Gumbel-max draw. It supports contiguous, paged, and CPU-prefix resident decode.
Top-k, active top-p, active typical-p, active top-n-sigma, penalties, logit bias,
and `min_keep > 1` decline this sampling tail before it mutates resident KV;
the general sampler handles them.
`min_p = 0` or an omitted min-p retains the existing temperature-only path.

The filter compares probability ratios instead of adding `log(min_p)` to a
large logit. Thresholds close enough for floating-point rounding to change the
retained set, `min_p = 1`, tiny normalized thresholds, and invalid arithmetic
return a sentinel. The caller reads the already-computed logits and uses the CPU
sampler with an advancing draw position. It does not replay the forward pass or
append the same KV row twice. Errors after a resident append propagate.

Temperature division precedes centering, matching the CPU sampler's floating-point
order. Centering precedes the Gumbel addition so large common offsets cannot
erase the random draw. The shared uniform helper clamps the single 24-bit
endpoint that previously rounded to one; every other uniform is unchanged.
GPU Gumbel draws and CPU cumulative-probability draws are different seeded
algorithms, so their token sequences need not match.

The boundary margin is a conservative guard, not a proof of identical math across
all CPU and CUDA libraries. Device qualification must compare complete retained
sets as well as sampled distributions before enabling this path by default.

## Portable and Mac checks

```sh
cargo test --lib resident_min_p
cargo test --lib resident_cuda_min_p_sampling_requires_an_explicit_opt_in
cargo test --lib resident_sampling_completion
# Requires clang++ or a C++17 compiler selected with CXX; does not use CUDA.
cargo test --lib cuda_min_p_host_classifier_matches_cpu_or_falls_back -- --ignored --nocapture
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
```

The explicit host check compiles the production C++ helpers, compares retained
sets with the Rust CPU sampler, and exhausts all 16,777,216 possible uniform
mantissas. It checks temperature scaling at large offsets and parses all kernel
bodies with host syntax stubs. These checks do not validate NVRTC compilation or
CUDA execution.

## Required NVIDIA device checks

Run on a supported NVIDIA host with its driver and NVRTC available:

```sh
CAMELID_REQUIRE_CUDA_TESTS=1 cargo test --features cuda --lib min_p_sampler -- --ignored --nocapture
CAMELID_REQUIRE_CUDA_TESTS=1 cargo test --features cuda --lib sampling_uniform_endpoint_stays_finite_in_both_kernels -- --ignored --nocapture
CAMELID_REQUIRE_CUDA_TESTS=1 cargo test --features cuda --lib gumbel_sampler_matches_stateless_reference -- --ignored --nocapture
```

Required mode fails if the device or kernel compilation is unavailable. The tests
cover complete masks, block-size boundaries, normalized rounding boundaries,
equal maxima, invalid rows, seeded reproducibility, temperature-dependent
distributions, large offsets, and the uniform endpoint. Follow these with model
decode checks with `CAMELID_CUDA_MIN_P_SAMPLING=1` in contiguous, paged, and
CPU-prefix modes. Include repeated seeded requests, candidate floors and combined
filters compared with the general sampler, and a boundary fallback followed by
continuation that verifies exactly one KV row was consumed. Record the GPU,
driver, NVRTC version, and model hash with the results. No throughput or
end-to-end NVIDIA validation is implied by the portable checks.
