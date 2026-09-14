// Opt-in packed-fragment projection backend for the isolated FP16 diagnostic.
// Included only by metal_fp16_probe.rs; serving and ordinary verifier routes
// never consult this flag or consume this weight representation.

struct Fp16SgKernels {
    dequant_q4: ComputePipelineState,
    dequant_q6: ComputePipelineState,
    pack_f32: ComputePipelineState,
    nt1: ComputePipelineState,
    nt2: ComputePipelineState,
    nt4: ComputePipelineState,
    nt8: ComputePipelineState,
}

fn fp16_sg_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CAMELID_BENCH_FP16_SG")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

fn fp16_sg_kernels(k: &MetalLinearKernel) -> Option<&'static Fp16SgKernels> {
    static KERNELS: OnceLock<Option<Fp16SgKernels>> = OnceLock::new();
    KERNELS
        .get_or_init(|| {
            // Match the original canonical dequantizer's compiler options. Only
            // its final half address is changed by the fused packing shader.
            let stage_options = CompileOptions::new();
            let stage_source = [
                LINEAR_ROW_SHADER,
                "\n",
                include_str!("metal_fp16_sg_stage.metal"),
            ]
            .concat();
            let stage_library = k
                .device
                .new_library_with_source(&stage_source, &stage_options)
                .map_err(|err| eprintln!("[fp16-sg] staging compile failed: {err}"))
                .ok()?;
            let options = CompileOptions::new();
            options.set_fast_math_enabled(false);
            let library = k
                .device
                .new_library_with_source(include_str!("metal_fp16_sg.metal"), &options)
                .map_err(|err| eprintln!("[fp16-sg] projection compile failed: {err}"))
                .ok()?;
            let pipeline =
                |library: &metal::LibraryRef, name: &str| -> Option<ComputePipelineState> {
                    let function = library.get_function(name, None).ok()?;
                    let p = k
                        .device
                        .new_compute_pipeline_state_with_function(&function)
                        .ok()?;
                    admitted_32_lane_pipeline(Some(&p))?;
                    Some(p)
                };
            Some(Fp16SgKernels {
                dequant_q4: pipeline(&stage_library, "fp16_sg_q4k_dequant_pack")?,
                dequant_q6: pipeline(&stage_library, "fp16_sg_q6k_dequant_pack")?,
                pack_f32: pipeline(&stage_library, "fp16_sg_pack_activation_f32")?,
                nt1: pipeline(&library, "sg_fragment_nt1")?,
                nt2: pipeline(&library, "sg_fragment_nt2")?,
                nt4: pipeline(&library, "sg_fragment_nt4")?,
                nt8: pipeline(&library, "sg_fragment_nt8")?,
            })
        })
        .as_ref()
}

fn fp16_sg_physical_rows(rows: usize) -> usize {
    match rows {
        1..=8 => 8,
        9..=16 => 16,
        17..=32 => 32,
        33..=64 => 64,
        _ => 0,
    }
}

/// Pack from the original f32 activation directly, then project. All physical
/// B cells that the selected NT kernel reads are written, including zero pads.
/// A is already packed during one-time canonical dequantization. All target
/// projection output counts are multiples of eight; no padded A tail exists.
#[allow(clippy::too_many_arguments)]
fn encode_fp16_sg_projection(
    e: &metal::ComputeCommandEncoderRef,
    k: &MetalLinearKernel,
    sg: &Fp16SgKernels,
    keep: &mut Vec<Buffer>,
    input: &Buffer,
    weight: &Buffer,
    panel: &Buffer,
    out: &Buffer,
    width: usize,
    rows: usize,
    n: usize,
) {
    let physical = fp16_sg_physical_rows(n);
    assert!(physical > 0 && width.is_multiple_of(256) && rows.is_multiple_of(8));
    assert!(
        input.length() >= (n * width * 4) as u64 && panel.length() >= (physical * width * 2) as u64
    );
    assert!(weight.length() >= (rows * width * 2) as u64 && out.length() >= (n * rows * 4) as u64);
    let scalar = pool_get(k, 16);
    let fields = [width as u32, rows as u32, n as u32, physical as u32];
    unsafe {
        std::ptr::copy_nonoverlapping(fields.as_ptr(), scalar.contents().cast::<u32>(), 4);
    }
    e.set_compute_pipeline_state(&sg.pack_f32);
    e.set_buffer(0, Some(input), 0);
    e.set_buffer(1, Some(panel), 0);
    e.set_buffer(2, Some(&scalar), 0);
    e.set_buffer(3, Some(&scalar), 8);
    e.set_buffer(4, Some(&scalar), 12);
    dispatch_1d(e, &sg.pack_f32, physical * width);
    let pipeline = match physical {
        8 => &sg.nt1,
        16 => &sg.nt2,
        32 => &sg.nt4,
        64 => &sg.nt8,
        _ => unreachable!(),
    };
    e.set_compute_pipeline_state(pipeline);
    e.set_buffer(0, Some(weight), 0);
    e.set_buffer(1, Some(panel), 0);
    e.set_buffer(2, Some(out), 0);
    e.set_buffer(3, Some(&scalar), 0);
    e.set_buffer(4, Some(&scalar), 4);
    e.set_buffer(5, Some(&scalar), 8);
    e.dispatch_thread_groups(
        metal::MTLSize {
            width: (rows / 8) as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    keep.push(scalar);
}
