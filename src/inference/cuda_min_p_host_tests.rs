//! Portable checks for the scalar math shared with the CUDA min-p kernel.
//! These execute the production C++ helpers, but do not exercise a CUDA device.

use super::*;
use std::fmt::Write as _;
use std::process::Command;

struct Case {
    name: String,
    logits: Vec<f32>,
    min_p: f32,
    must_fallback: bool,
}

fn case(name: &str, logits: Vec<f32>, min_p: f32, must_fallback: bool) -> Case {
    Case {
        name: name.to_string(),
        logits,
        min_p,
        must_fallback,
    }
}

fn next_random(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 32) as u32
}

#[test]
#[ignore = "requires a host C++17 compiler; does not require CUDA"]
fn cuda_min_p_host_classifier_matches_cpu_or_falls_back() {
    let mut cases = vec![
        // Normalizing both probabilities can change an inclusive f32 boundary.
        case(
            "normalized inclusive boundary",
            vec![0.0, -0.1],
            f32::from_bits(0x3f67_a36e),
            true,
        ),
        case("equal maxima", vec![2.0, 2.0, -1.0], 1.0, true),
        case(
            "rounded unequal maxima",
            vec![0.0, -2.0f32.powi(-26)],
            1.0,
            true,
        ),
        case("signed zero maxima", vec![0.0, -0.0, -1.0], 1.0, true),
        // The normalized threshold rounds to zero, retaining a zero weight.
        case(
            "subnormal threshold",
            vec![0.0, 0.0, -110.0],
            f32::from_bits(1),
            true,
        ),
        case(
            "small normal threshold",
            vec![0.0, -90.0],
            f32::MIN_POSITIVE,
            true,
        ),
        // Adding ln(min_p) to this maximum loses significant threshold bits.
        case(
            "large common offset",
            vec![100_000_000.0, 99_999_992.0],
            0.0004,
            false,
        ),
        case("single candidate", vec![7.0], 0.25, false),
        case("negative logits", vec![-70.0, -72.0, -75.0], 0.05, false),
        case("positive infinity", vec![0.0, f32::INFINITY], 0.1, true),
        case("negative infinity", vec![0.0, f32::NEG_INFINITY], 0.1, true),
        case("nan", vec![0.0, f32::NAN], 0.1, true),
    ];
    for n in [1, 255, 256, 257, 513] {
        let mut logits: Vec<f32> = (0..n).map(|i| -((i % 31) as f32)).collect();
        logits[n - 1] = 2.0;
        cases.push(case(&format!("vocabulary {n}"), logits, 0.125, false));
    }

    let mut random = 0x915e_55a9_da3c_1891_u64;
    for row in 0..160 {
        let n = [1, 2, 3, 7, 255, 256, 257, 513][row % 8];
        let offset = [0.0, -100.0, 100.0, 100_000_000.0][row % 4];
        let logits: Vec<f32> = (0..n)
            .map(|_| offset + (next_random(&mut random) % 12_001) as f32 / 100.0 - 60.0)
            .collect();
        let probability_bits =
            0x0080_0000 + next_random(&mut random) % (1.0f32.to_bits() - 0x0080_0000);
        cases.push(case(
            &format!("random row {row}"),
            logits.clone(),
            f32::from_bits(probability_bits),
            false,
        ));
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let selected = next_random(&mut random) as usize % n;
        let ratio = (logits[selected] - maximum).exp();
        for delta in -2_i64..=2 {
            let bits =
                (i64::from(ratio.to_bits()) + delta).clamp(1, i64::from(1.0f32.to_bits())) as u32;
            cases.push(case(
                &format!("random boundary {row}/{delta}"),
                logits.clone(),
                f32::from_bits(bits),
                false,
            ));
        }
    }

    let directory = tempfile::tempdir().expect("create C++ test directory");
    let source = directory.path().join("sampling_math.cpp");
    let syntax_source = directory.path().join("sampling_kernel_syntax.cpp");
    let executable = directory.path().join(if cfg!(windows) {
        "sampling_math.exe"
    } else {
        "sampling_math"
    });
    std::fs::write(&source, HOST_TEST_SOURCE).expect("write C++ math test");
    std::fs::write(&syntax_source, KERNEL_SYNTAX_SOURCE).expect("write C++ kernel syntax test");
    // MSVC's flags differ from GCC/Clang's. Run from a Developer Command
    // Prompt on MSVC, or set CXX to a compiler executable (not shell flags).
    let compiler = std::env::var_os("CXX").unwrap_or_else(|| {
        if cfg!(target_env = "msvc") {
            "cl".into()
        } else {
            "clang++".into()
        }
    });
    let msvc = std::path::Path::new(&compiler)
        .file_stem()
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|name| {
            name.eq_ignore_ascii_case("cl") || name.eq_ignore_ascii_case("clang-cl")
        });
    let compile = |source: &std::path::Path, output: Option<&std::path::Path>| {
        let mut command = Command::new(&compiler);
        // MSVC also writes an object file; keep all generated files temporary.
        command.current_dir(directory.path());
        let include = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/cuda_resident");
        if msvc {
            command
                .args(["/nologo", "/std:c++17", "/EHsc", "/fp:strict"])
                .arg("/I")
                .arg(include);
            if let Some(output) = output {
                command.arg("/O2").arg("/Fe:").arg(output);
            } else {
                command.arg("/Zs");
            }
        } else {
            command
                .args(["-std=c++17", "-ffp-contract=off", "-fno-fast-math"])
                .arg("-I")
                .arg(include);
            if let Some(output) = output {
                command.arg("-O2").arg("-o").arg(output);
            } else {
                command.arg("-fsyntax-only");
            }
        }
        let result = command
            .arg(source)
            .output()
            .expect("run host C++ compiler (set CXX; MSVC requires a Developer Command Prompt)");
        assert!(
            result.status.success(),
            "C++ compilation failed for {}: {}{}",
            source.display(),
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
    };
    compile(&source, Some(&executable));
    // Parse every production CUDA kernel body as C++ using syntax-only CUDA
    // declarations. This catches ordinary C++ errors on a host without NVRTC;
    // it does not validate CUDA compilation, synchronization, or execution.
    compile(&syntax_source, None);

    let mut input = String::new();
    let mut score_probes: Vec<(f32, f32, f32)> = Vec::new();
    for temperature in [0.25, 1.0, 2.0, 3.0] {
        for (logit, maximum) in [
            (99_999_992.0, 100_000_000.0),
            (100_000_000.0, 100_000_000.0),
            (-0.0, 0.0),
            (-72.0, -70.0),
            (-1.0, 1.0),
            (f32::MAX, f32::MAX),
            (-f32::MAX, f32::MAX),
        ] {
            score_probes.push((logit, maximum, temperature));
        }
    }
    writeln!(&mut input, "{}", score_probes.len()).unwrap();
    for (logit, maximum, temperature) in &score_probes {
        writeln!(
            &mut input,
            "{} {} {}",
            logit.to_bits(),
            maximum.to_bits(),
            temperature.to_bits()
        )
        .unwrap();
    }
    writeln!(&mut input, "{}", cases.len()).unwrap();
    for case in &cases {
        write!(&mut input, "{} {}", case.logits.len(), case.min_p.to_bits()).unwrap();
        for logit in &case.logits {
            write!(&mut input, " {}", logit.to_bits()).unwrap();
        }
        input.push('\n');
    }
    let input_path = directory.path().join("cases.txt");
    std::fs::write(&input_path, input).expect("write exact-bit cases");
    let output = Command::new(executable)
        .arg(input_path)
        .output()
        .expect("execute C++ math test");
    assert!(
        output.status.success(),
        "C++ math test failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("C++ output is UTF-8");
    let mut lines = stdout.lines();
    assert_eq!(lines.next(), Some("uniform 16777216 changed 1"));
    for (logit, maximum, temperature) in &score_probes {
        let bits: u32 = lines
            .next()
            .expect("missing centered-logit probe result")
            .parse()
            .expect("centered-logit float bits");
        let actual = f32::from_bits(bits);
        // The CPU sampler divides individual logits before subtracting the
        // scaled maximum. Centering first can change the distribution greatly.
        let expected = (*logit / *temperature) - (*maximum / *temperature);
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        if *logit == 99_999_992.0 && *maximum == 100_000_000.0 && *temperature == 3.0 {
            assert_eq!(
                actual, -4.0,
                "large-offset temperature scaling must match CPU"
            );
        }
    }
    let mut checked = 0;
    let mut fallbacks = 0;
    for case in cases {
        let classifications: Vec<i32> = lines
            .next()
            .unwrap_or_else(|| panic!("missing C++ result for {}", case.name))
            .split_whitespace()
            .map(|value| value.parse().expect("integer classification"))
            .collect();
        assert_eq!(classifications.len(), case.logits.len(), "{}", case.name);
        assert!(classifications.iter().all(|value| (-1..=1).contains(value)));
        let fallback = classifications.contains(&-1);
        assert!(
            !case.must_fallback || fallback,
            "{} must fall back",
            case.name
        );
        if fallback {
            fallbacks += 1;
        }
        let logits =
            CpuTensor::from_f32("logits", vec![1, case.logits.len()], case.logits.clone()).unwrap();
        for temperature in [0.25, 1.0, 2.0] {
            let config = SamplingConfig {
                temperature,
                min_p: Some(case.min_p),
                ..SamplingConfig::default()
            };
            let expected = sampling_weights_with_config(&logits, &config, &[]);
            if !case.logits.iter().all(|logit| logit.is_finite()) {
                assert!(expected.is_err(), "{}", case.name);
                assert!(fallback, "{}", case.name);
                continue;
            }
            let expected = expected.unwrap();
            if !fallback {
                let mut mask = vec![false; case.logits.len()];
                for (token, _) in expected {
                    mask[token] = true;
                }
                let actual: Vec<bool> = classifications.iter().map(|value| *value == 1).collect();
                assert_eq!(actual, mask, "{} at temperature {temperature}", case.name);
                checked += 1;
            }
        }
    }
    assert!(lines.next().is_none(), "unexpected extra C++ output");
    assert!(checked > 100, "too few nonambiguous rows: {checked}");
    assert!(fallbacks > 50, "too few fallback rows: {fallbacks}");
    eprintln!("CUDA scalar host oracle: {checked} masks matched; {fallbacks} ambiguous/unsupported rows deferred; {} temperature probes matched; kernel C++ syntax parsed", score_probes.len());
}

const HOST_TEST_SOURCE: &str = r#"
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
#define CAMELID_SAMPLING_HOST_TEST 1
#define __device__
#define __forceinline__ inline
using std::isfinite;
#include "sampling.cuh"

float from_bits(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::uint32_t to_bits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    unsigned changed = 0;
    for (unsigned m = 0; m < 16777216u; ++m) {
        float old_value = (static_cast<float>(m) + 0.5f) / 16777216.0f;
        float value = sampling_uniform_from_u24(m);
        float noise = -logf(-logf(value));
        if (!(value > 0.0f && value < 1.0f && std::isfinite(noise))) return 2;
        if (value != old_value) {
            if (old_value != 1.0f || m != 16777215u) return 3;
            ++changed;
        }
    }
    std::cout << "uniform 16777216 changed " << changed << '\n';
    std::ifstream input(argv[1]);
    unsigned probes;
    if (!(input >> probes)) return 4;
    for (unsigned probe = 0; probe < probes; ++probe) {
        std::uint32_t logit, maximum, temperature;
        if (!(input >> logit >> maximum >> temperature)) return 4;
        std::cout << to_bits(sampling_centered_logit(
            from_bits(logit), from_bits(maximum), from_bits(temperature))) << '\n';
    }
    unsigned rows;
    if (!(input >> rows)) return 4;
    for (unsigned row = 0; row < rows; ++row) {
        int n;
        std::uint32_t p_bits;
        if (!(input >> n >> p_bits) || n < 1) return 5;
        std::vector<float> logits(n);
        float maximum = -std::numeric_limits<float>::infinity();
        for (float& logit : logits) {
            std::uint32_t bits;
            if (!(input >> bits)) return 6;
            logit = from_bits(bits);
            maximum = fmaxf(maximum, logit);
        }
        for (int i = 0; i < n; ++i) {
            if (i) std::cout << ' ';
            std::cout << min_p_candidate(logits[i], maximum, from_bits(p_bits), n);
        }
        std::cout << '\n';
    }
    return 0;
}
"#;

const KERNEL_SYNTAX_SOURCE: &str = r#"
#include <cmath>
#define __device__
#define __forceinline__ inline
#define __global__
#define __shared__
#ifdef _MSC_VER
#define __restrict__ __restrict
#endif
#define CAMELID_CUDA_SAMPLING_TESTS 1
using std::isfinite;
struct NativeCudaDim { unsigned int x, y, z; };
extern NativeCudaDim threadIdx, blockIdx, blockDim, gridDim;
void __syncthreads();
float __int_as_float(int);
#include "sampling.cuh"
"#;
