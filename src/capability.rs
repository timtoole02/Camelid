//! Runtime hardware capability detection.
//!
//! Probed once at startup and logged, so every performance tunable (VRAM-driven
//! resident-context sizing, thread-pool width, SIMD path selection, speculative
//! decoding admission) derives from the *detected* machine rather than being
//! hardcoded. CPU-only hosts, modest GPUs, and large-VRAM cards all read the same
//! struct and scale themselves from it.

use crate::cuda;

/// CPU SIMD instruction-set availability (runtime-detected, not compile-time).
#[derive(Debug, Clone, Default)]
pub struct SimdCaps {
    pub avx2: bool,
    pub avx512f: bool,
    pub fma: bool,
    pub neon: bool,
}

impl SimdCaps {
    /// Short human label of the widest available SIMD, for logs.
    pub fn label(&self) -> &'static str {
        if self.avx512f {
            "AVX-512"
        } else if self.avx2 {
            "AVX2"
        } else if self.neon {
            "NEON"
        } else {
            "scalar"
        }
    }
}

/// A snapshot of the host's inference-relevant capabilities.
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub cuda_available: bool,
    pub cuda_device_count: usize,
    pub cuda_device_name: Option<String>,
    pub cuda_compute_capability: Option<(u32, u32)>,
    /// Tensor cores are present from compute capability 7.0 (Volta) onward.
    pub cuda_tensor_cores: bool,
    pub cuda_vram_total_bytes: u64,
    pub cuda_vram_free_bytes: u64,
    pub cpu_logical_cores: usize,
    pub host_ram_total_bytes: u64,
    pub host_ram_free_bytes: u64,
    pub simd: SimdCaps,
}

impl HardwareProfile {
    /// Probe the machine. Cheap and side-effect-free: the CUDA probe opens a
    /// context and reads device attributes but compiles no kernels, and degrades
    /// to "no CUDA" cleanly on hosts without a device.
    pub fn detect() -> Self {
        let cap = cuda::probe_capability();
        let (
            cuda_available,
            cuda_device_count,
            cuda_device_name,
            cuda_compute_capability,
            cuda_tensor_cores,
            cuda_vram_total_bytes,
            cuda_vram_free_bytes,
        ) = match &cap {
            Some(c) => (
                true,
                c.device_count,
                Some(c.device_name.clone()),
                Some(c.compute_capability),
                c.compute_capability.0 >= 7,
                c.vram_total_bytes,
                c.vram_free_bytes,
            ),
            None => (false, 0, None, None, false, 0, 0),
        };
        let cpu_logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let (host_ram_total_bytes, host_ram_free_bytes) = host_ram_bytes();
        HardwareProfile {
            cuda_available,
            cuda_device_count,
            cuda_device_name,
            cuda_compute_capability,
            cuda_tensor_cores,
            cuda_vram_total_bytes,
            cuda_vram_free_bytes,
            cpu_logical_cores,
            host_ram_total_bytes,
            host_ram_free_bytes,
            simd: detect_simd(),
        }
    }

    /// Process-wide cached probe. The first call probes the machine (opening a
    /// CUDA context once); every later call reuses the same snapshot. Use this on
    /// non-hot paths (e.g. the catalog handler) that want the profile without
    /// re-probing the device on every request.
    pub fn cached() -> &'static HardwareProfile {
        static CACHED: std::sync::OnceLock<HardwareProfile> = std::sync::OnceLock::new();
        CACHED.get_or_init(HardwareProfile::detect)
    }

    /// Emit the detected profile to stderr, in the same `[hw]` voice as the other
    /// startup diagnostics. This is the line every tunable is justified against.
    pub fn log(&self) {
        const GIB: f64 = (1024 * 1024 * 1024) as f64;
        if self.cuda_available {
            let (cc_major, cc_minor) = self.cuda_compute_capability.unwrap_or((0, 0));
            eprintln!(
                "[hw] GPU: {} (x{}) | compute {}.{} | tensor-cores {} | VRAM {:.1} GiB free / {:.1} GiB total",
                self.cuda_device_name.as_deref().unwrap_or("unknown"),
                self.cuda_device_count,
                cc_major,
                cc_minor,
                if self.cuda_tensor_cores { "yes" } else { "no" },
                self.cuda_vram_free_bytes as f64 / GIB,
                self.cuda_vram_total_bytes as f64 / GIB,
            );
        } else {
            eprintln!("[hw] GPU: none detected — CPU backend is the inference path");
        }
        eprintln!(
            "[hw] CPU: {} logical cores | SIMD {} (avx2={} avx512f={} fma={} neon={}) | RAM {:.1} GiB free / {:.1} GiB total",
            self.cpu_logical_cores,
            self.simd.label(),
            self.simd.avx2,
            self.simd.avx512f,
            self.simd.fma,
            self.simd.neon,
            self.host_ram_free_bytes as f64 / GIB,
            self.host_ram_total_bytes as f64 / GIB,
        );
    }
}

fn detect_simd() -> SimdCaps {
    #[cfg(target_arch = "x86_64")]
    {
        SimdCaps {
            avx2: std::is_x86_feature_detected!("avx2"),
            avx512f: std::is_x86_feature_detected!("avx512f"),
            fma: std::is_x86_feature_detected!("fma"),
            neon: false,
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        SimdCaps {
            avx2: false,
            avx512f: false,
            fma: false,
            neon: std::arch::is_aarch64_feature_detected!("neon"),
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        SimdCaps::default()
    }
}

/// (total, available) physical RAM in bytes. Per-OS, best-effort; returns (0, 0)
/// on platforms without a probe so callers treat it as "unknown".
#[cfg(windows)]
fn host_ram_bytes() -> (u64, u64) {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: zero-initialized MEMORYSTATUSEX with dwLength set, as the API requires.
    unsafe {
        let mut status: MEMORYSTATUSEX = std::mem::zeroed();
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if GlobalMemoryStatusEx(&mut status) != 0 {
            (status.ullTotalPhys, status.ullAvailPhys)
        } else {
            (0, 0)
        }
    }
}

#[cfg(target_os = "linux")]
fn host_ram_bytes() -> (u64, u64) {
    // /proc/meminfo reports kB; MemAvailable is the kernel's own estimate of what
    // is reclaimable for new allocations (better than MemFree for our purposes).
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
        return (0, 0);
    };
    let field = |key: &str| -> u64 {
        text.lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .map(|kb| kb * 1024)
            .unwrap_or(0)
    };
    (field("MemTotal:"), field("MemAvailable:"))
}

#[cfg(target_os = "macos")]
fn host_ram_bytes() -> (u64, u64) {
    // Reuse the Mach-backed probe that also protects KV-cache admission so the
    // catalog fit advisor and runtime safety gate agree on host-memory capacity.
    crate::gait::host_ram_status().unwrap_or((0, 0))
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
fn host_ram_bytes() -> (u64, u64) {
    (0, 0)
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::host_ram_bytes;

    #[test]
    fn host_ram_probe_reports_live_physical_ram() {
        let (total, available) = host_ram_bytes();
        assert!(total > 0, "total physical RAM must be positive");
        assert!(available > 0, "available physical RAM must be positive");
        assert!(
            available <= total,
            "available {available} must not exceed total {total}"
        );
    }
}
