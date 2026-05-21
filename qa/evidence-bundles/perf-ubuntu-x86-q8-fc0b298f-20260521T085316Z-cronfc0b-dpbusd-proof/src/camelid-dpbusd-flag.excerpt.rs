}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_q8_packed_rows4_avx2_dot_enabled() -> bool {
    #[cfg(test)]
    {
        q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PACKED_ROWS4_AVX2_DOT")
    }
    #[cfg(not(test))]
    {
        static X86_Q8_PACKED_ROWS4_AVX2_DOT_ENABLED: OnceLock<bool> = OnceLock::new();
        *X86_Q8_PACKED_ROWS4_AVX2_DOT_ENABLED.get_or_init(|| {
            q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PACKED_ROWS4_AVX2_DOT")
        })
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_q8_packed_rows4_avx512vnni_dpwssd_dot_enabled() -> bool {
    #[cfg(test)]
    {
        q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PACKED_ROWS4_AVX512VNNI_DPWSSD_DOT")
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(test))]
    {
        static X86_Q8_PACKED_ROWS4_AVX512VNNI_DPWSSD_DOT_ENABLED: OnceLock<bool> = OnceLock::new();
        *X86_Q8_PACKED_ROWS4_AVX512VNNI_DPWSSD_DOT_ENABLED.get_or_init(|| {
            q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PACKED_ROWS4_AVX512VNNI_DPWSSD_DOT")
                && std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vnni")
        })
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_q8_packed_rows4_avx512vnni_dpbusd_dot_enabled() -> bool {
    #[cfg(test)]
    {
        q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PACKED_ROWS4_AVX512VNNI_DPBUSD_DOT")
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(test))]
    {
        static X86_Q8_PACKED_ROWS4_AVX512VNNI_DPBUSD_DOT_ENABLED: OnceLock<bool> = OnceLock::new();
        *X86_Q8_PACKED_ROWS4_AVX512VNNI_DPBUSD_DOT_ENABLED.get_or_init(|| {
            q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PACKED_ROWS4_AVX512VNNI_DPBUSD_DOT")
                && std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vnni")
        })
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_q8_packed_rows4_avx2_dot_hoist_enabled() -> bool {
    #[cfg(test)]
    {
        q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PACKED_ROWS4_AVX2_DOT_HOIST")
