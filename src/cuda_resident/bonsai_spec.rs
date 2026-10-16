//! Qwen3.5 speculative verification keeps its recurrent rollback state on CUDA.
//! Attention rows beyond the restored watermark are overwritten on replay and
//! never attended, so copying the entire KV cache would only waste VRAM/bandwidth.
use super::*;

pub(crate) struct Qwen35StateSnapshot {
    conv: Vec<CudaSlice<f32>>,
    recurrent: Vec<CudaSlice<f32>>,
    filled: usize,
}

impl CudaResidentDecode {
    pub(crate) fn snapshot_qwen35_state(&self) -> Result<Qwen35StateSnapshot, String> {
        if self.qwen35.is_none() {
            return Err("Qwen3.5 snapshot requested for another architecture".into());
        }
        let copy = |buffers: &[CudaSlice<f32>]| {
            buffers
                .iter()
                .map(|buffer| {
                    self.k
                        .stream
                        .clone_dtod(buffer)
                        .map_err(|error| format!("Qwen3.5 snapshot: {error}"))
                })
                .collect::<Result<Vec<_>, String>>()
        };
        Ok(Qwen35StateSnapshot {
            conv: copy(&self.ssm_conv_state)?,
            recurrent: copy(&self.ssm_state)?,
            filled: self.filled,
        })
    }

    pub(crate) fn refresh_qwen35_state(
        &self,
        snapshot: &mut Qwen35StateSnapshot,
    ) -> Result<(), String> {
        let copy = |source: &[CudaSlice<f32>], destination: &mut [CudaSlice<f32>]| {
            if source.len() != destination.len()
                || source
                    .iter()
                    .zip(destination.iter())
                    .any(|(a, b)| a.len() != b.len())
            {
                return Err("Qwen3.5 snapshot shape does not match this graph".to_string());
            }
            for (source, destination) in source.iter().zip(destination.iter_mut()) {
                self.k
                    .stream
                    .memcpy_dtod(source, destination)
                    .map_err(|error| format!("Qwen3.5 snapshot refresh: {error}"))?;
            }
            Ok(())
        };
        copy(&self.ssm_conv_state, &mut snapshot.conv)?;
        copy(&self.ssm_state, &mut snapshot.recurrent)?;
        snapshot.filled = self.filled;
        Ok(())
    }

    pub(crate) fn restore_qwen35_state(
        &mut self,
        snapshot: &Qwen35StateSnapshot,
    ) -> Result<(), String> {
        let stream = self.k.stream.clone();
        let copy = |source: &[CudaSlice<f32>], destination: &mut [CudaSlice<f32>]| {
            if source.len() != destination.len()
                || source
                    .iter()
                    .zip(destination.iter())
                    .any(|(a, b)| a.len() != b.len())
            {
                return Err("Qwen3.5 snapshot shape does not match this graph".to_string());
            }
            for (source, destination) in source.iter().zip(destination.iter_mut()) {
                stream
                    .memcpy_dtod(source, destination)
                    .map_err(|error| format!("Qwen3.5 snapshot restore: {error}"))?;
            }
            Ok(())
        };
        copy(&snapshot.conv, &mut self.ssm_conv_state)?;
        copy(&snapshot.recurrent, &mut self.ssm_state)?;
        self.set_filled(snapshot.filled);
        Ok(())
    }

    /// Materialize real device state for the hardware qualification test. Every
    /// float is compared by its bits, including SSM/conv state and live KV rows.
    #[cfg(test)]
    pub(crate) fn qwen35_state_bits(&self) -> Result<Vec<u32>, String> {
        let mut bits = vec![self.filled as u32];
        for buffer in self.ssm_conv_state.iter().chain(self.ssm_state.iter()) {
            let values = self
                .k
                .stream
                .clone_dtoh(buffer)
                .map_err(|error| format!("Qwen3.5 state readback: {error}"))?;
            bits.extend(values.into_iter().map(f32::to_bits));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            if matches!(&layer.kind, LayerKind::Full) && self.filled > 0 {
                let (keys, values) = self.read_kv_layer(index, self.filled)?;
                bits.extend(keys.into_iter().chain(values).map(f32::to_bits));
            }
        }
        Ok(bits)
    }

    /// Verify a short consecutive proposal chain using shared packed weight reads.
    /// The scalar and batched projections use the same arithmetic; only the target
    /// sampler sees these rows. Streamed layers reuse one transfer for the chain.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn verify_qwen35_batch_logits(
        &mut self,
        embeddings: &[f32],
        cos_all: &[f32],
        sin_all: &[f32],
        base_position: usize,
        k: usize,
        scale: f32,
    ) -> Result<Vec<Vec<f32>>, String> {
        if k == 0 || k > MAX_VERIFY_K {
            return Err(format!("verify_batch: k={k} out of 1..={MAX_VERIFY_K}"));
        }
        if self.qwen35.is_none()
            || !self.output_quant.supports_batched()
            || !self.layers.iter().all(|layer| {
                layer.quants.iter().all(|quant| quant.supports_batched())
                    && match &layer.kind {
                        LayerKind::Full => true,
                        LayerKind::Ssm(ssm) => {
                            ssm.quants.iter().all(|quant| quant.supports_batched())
                        }
                    }
            })
        {
            return Err("verify_batch: model has no resident batched projection path".into());
        }
        if k > self.batched_verify_token_cap() {
            return Err(format!(
                "verify_batch: k={k} exceeds this quant lane's shared-memory cap of {}",
                self.batched_verify_token_cap()
            ));
        }
        if base_position
            .checked_add(k)
            .is_none_or(|end| end > self.max_pos)
        {
            return Err("CUDA draft verification exceeds the resident context capacity".into());
        }
        let map = |e: cudarc::driver::DriverError| format!("cuda draft verify: {e}");
        // The per-layer batched stack lives in `run_batched_layer_stack` (shared with
        // batched prefill); here we only need the dims for input staging and the final
        // logits projection.
        let (hidden, vocab, eps) = (self.hidden, self.vocab, self.eps);
        let half = self.rope_dim / 2;
        if embeddings.len() < k * hidden || cos_all.len() < k * half || sin_all.len() < k * half {
            return Err("verify_batch: input slices too short".into());
        }
        self.ensure_verify_scratch()?;
        let s = self.k.stream.clone();
        let mut sc = self.verify_scratch.take().expect("allocated above");

        s.memcpy_htod(
            &embeddings[..k * hidden],
            &mut sc.vh.slice_mut(0..k * hidden),
        )
        .map_err(map)?;
        s.memcpy_htod(&cos_all[..k * half], &mut sc.vcos.slice_mut(0..k * half))
            .map_err(map)?;
        s.memcpy_htod(&sin_all[..k * half], &mut sc.vsin.slice_mut(0..k * half))
            .map_err(map)?;

        self.run_batched_layer_stack(&mut sc, &s, base_position, k, scale, false)?;
        let output_bmma = if self.k.fast_q1
            && self.output_quant == ProjQuant::Q1_0
            && !prism_bmma_shape_enabled(&self.k, hidden, k)
        {
            launch_prism_rms_norm_q8_batched(
                &s,
                &self.k.prism_rms_norm_q8_batched,
                &sc.vh,
                &self.final_norm,
                &mut sc.viq,
                &mut sc.vis,
                hidden,
                eps,
                k,
            )
            .map_err(map)?;
            false
        } else {
            launch_rms_norm_batched(
                &s,
                &self.k.rms_norm_batched,
                &sc.vh,
                &self.final_norm,
                &mut sc.vn,
                hidden,
                eps,
                k,
            )
            .map_err(map)?;
            quantize_batched_for_lanes(
                &s,
                &self.k,
                &sc.vn,
                &mut sc.viq,
                &mut sc.vis,
                &mut sc.vibits,
                &mut sc.vibscales,
                &mut sc.viqk,
                &mut sc.visk,
                hidden,
                k,
                std::slice::from_ref(&self.output_quant),
            )
            .map_err(map)?
        };
        dispatch_gemm_batched(
            &s,
            &self.k,
            self.output_quant,
            &sc.vn,
            &sc.vis,
            &sc.viq,
            &sc.vibits,
            &sc.vibscales,
            output_bmma,
            &sc.visk,
            &sc.viqk,
            &self.output_weight.as_view(),
            vocab,
            hidden,
            k,
            &mut sc.vlogits,
            0,
        )
        .map_err(map)?;
        let mut logits = vec![0.0f32; k * vocab];
        s.memcpy_dtoh(&sc.vlogits.slice(0..k * vocab), &mut logits)
            .map_err(map)?;
        self.k.ctx.synchronize().map_err(map)?;
        self.verify_scratch = Some(sc);
        self.set_filled(base_position + k);
        Ok(logits.chunks_exact(vocab).map(<[f32]>::to_vec).collect())
    }
}
