# Opt-in CUDA min-p sampling

`CAMELID_CUDA_MIN_P_SAMPLING=1` enables a resident CUDA sampling tail for
`0 < min_p <= 1` with the default candidate floor (`min_keep` absent or one).
It requires the existing CUDA resident decode and temperature sampling
lanes. The new tail is disabled by default pending full-model NVIDIA qualification.
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

## Portable checks

```sh
cargo test --lib resident_min_p
cargo test --lib resident_cuda_min_p_sampling_requires_an_explicit_opt_in
cargo test --lib resident_sampling_completion
# Requires a host C++17 compiler; does not use CUDA.
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

On Windows MSVC targets, run the host check from a Visual Studio Developer
PowerShell or Command Prompt; it uses `cl` with strict floating-point evaluation.
Other targets default to `clang++`. Set `CXX` to a compiler executable to override
the default; GCC-style compilers and `clang-cl` are also supported.

## Required NVIDIA device checks

Run on a supported NVIDIA host with its driver and NVRTC available:

```sh
CAMELID_REQUIRE_CUDA_TESTS=1 cargo test --features cuda --lib min_p_sampler -- --ignored --nocapture
CAMELID_REQUIRE_CUDA_TESTS=1 cargo test --features cuda --lib sampling_uniform_endpoint_stays_finite_in_both_kernels -- --ignored --nocapture
CAMELID_REQUIRE_CUDA_TESTS=1 cargo test --features cuda --lib gumbel_sampler_matches_stateless_reference -- --ignored --nocapture
CAMELID_REQUIRE_CUDA_TESTS=1 cargo test --features cuda --lib cuda_min_p_model_tails_preserve_fallback_kv_and_seeded_replay -- --ignored --nocapture --test-threads=1
```

Required mode fails if the device or kernel compilation is unavailable. The tests
cover complete masks through 128,256 tokens, block-size boundaries, normalized rounding boundaries,
equal maxima, invalid rows, seeded reproducibility, temperature-dependent
distributions, large offsets, and the uniform endpoint. Follow these with model
decode checks with `CAMELID_CUDA_MIN_P_SAMPLING=1` in contiguous, paged, and
CPU-prefix modes. Include repeated seeded requests, candidate floors and combined
filters compared with the general sampler, and a boundary fallback followed by
continuation that verifies exactly one KV row was consumed. Record the GPU,
driver, NVRTC version, and model hash with the results. No throughput or
end-to-end NVIDIA validation is implied by the portable checks.

The model-tail test runs a deterministic three-layer Q8 fixture in isolated
contiguous, paged, and CPU-prefix processes. It compares fallback logits and
complete KV history with ordinary forward execution, crosses a 16-token page
boundary, checks seeded replay through the public sampling entry point, and
verifies unsupported filters decline before touching the input token or KV.
This fixture gate does not replace qualification with a full GGUF model.

In PowerShell, set required mode before running the same Cargo commands without
their shell prefixes:

```powershell
$env:CAMELID_REQUIRE_CUDA_TESTS = '1'
cargo test --locked --features cuda --lib min_p_sampler -- --ignored --nocapture --test-threads=1
cargo test --locked --features cuda --lib sampling_uniform_endpoint_stays_finite_in_both_kernels -- --ignored --nocapture
cargo test --locked --features cuda --lib gumbel_sampler_matches_stateless_reference -- --ignored --nocapture
cargo test --locked --features cuda --lib cuda_min_p_model_tails_preserve_fallback_kv_and_seeded_replay -- --ignored --nocapture --test-threads=1
```

## NVIDIA fixture qualification (2026-10-12)

Windows x86-64, RTX 3060 Laptop GPU (6 GiB, compute capability 8.6), driver
576.83, CUDA toolkit 12.9.86:

- All six explicit sampler device tests passed with required CUDA mode,
  including NVRTC compilation, full retained masks through 128,256 tokens,
  seeded reproducibility, distributions, and the uniform endpoint.
- The three-layer Q8 fixture passed in all three decode modes: contiguous,
  paged, and CPU-prefix. Fallback preserved exact logits and KV history through
  continuation across a page boundary; seeded replay and filter admission passed.
- All 15 focused min-p, completion, device, and native-host tests passed. The
  MSVC host oracle exhausted all 16,777,216 uniform mantissas, matched 486
  retained-mask comparisons and 28 temperature probes, deferred 815 boundary or
  unsupported rows, and parsed every sampling kernel body.
- Formatting and diff checks passed.

These results qualify the kernels and deterministic fixture on this GPU. Full
GGUF request/decode qualification and throughput measurements remain pending;
the opt-in stays disabled by default.
