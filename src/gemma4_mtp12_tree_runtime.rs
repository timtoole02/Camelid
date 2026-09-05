//! Bounded W8 target tree integration. The ordinary linear path is unchanged.
use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_round_gate_preserves_budget_tail_and_checks_physical_capacity() {
        for (remaining, capacity, expected) in
            [(8, 8, false), (9, 8, true), (8, 7, false), (9, 7, false)]
        {
            let schedule = Gemma4Mtp12WidthScheduler::new_w8_padded_tail(8).unwrap();
            let plan = schedule
                .plan_round_with_capacity(remaining, capacity)
                .unwrap();
            assert_eq!(
                mtp12_tree_policy::round_eligible(
                    plan.logical_verify_width,
                    plan.physical_verify_width,
                    529,
                    529 + capacity,
                ),
                expected,
                "remaining={remaining} physical_capacity={capacity}"
            );
        }
        assert!(!mtp12_tree_policy::round_eligible(
            8,
            8,
            usize::MAX - 7,
            usize::MAX
        ));
        assert!(!mtp12_tree_policy::round_eligible(7, 8, 529, 2048));
        assert!(!mtp12_tree_policy::round_eligible(8, 16, 529, 2048));
    }

    fn pending_tree(state: &mut Gemma4DenseVerifierState) -> Gemma4PendingVerifierBatch {
        let ticket = state.record_completed_batch(128, 8).unwrap();
        let pending = state.pending.as_mut().unwrap();
        pending.tree_parents = Some([-1, 0, 1, 2, 3, 1, 5, 6]);
        assert_eq!(pending.ticket, ticket);
        *pending
    }

    #[test]
    fn tree_ticket_rejects_wrong_consumers_and_supports_abort() {
        let mut state = Gemma4DenseVerifierState {
            lane: Gemma4DenseSequenceLane::OrderedVerifier,
            logical_len: 128,
            next_ticket: 7,
            pending: None,
        };
        let pending = pending_tree(&mut state);
        for rows in 1..=8 {
            assert!(!pending.permits_linear_commit(pending.ticket, rows));
        }
        assert!(!pending.permits_linear_commit(pending.ticket + 1, 0));
        for path in [
            &[][..],
            &[1][..],
            &[0, 5][..],
            &[0, 1, 1][..],
            &[0, 1, 8][..],
        ] {
            assert!(!pending.permits_tree_path(pending.ticket, path));
            assert_eq!(state.logical_len, 128);
            assert_eq!(state.pending, Some(pending));
        }
        assert!(!pending.permits_tree_path(pending.ticket + 1, &[0, 1, 5]));
        assert!(pending.permits_linear_commit(pending.ticket, 0));
        assert_eq!(state.resolve_prefix(pending.ticket, 0), Some(128));
        assert_eq!(state.resolve_prefix(pending.ticket, 0), None);
        let linear = state.record_completed_batch(128, 4).unwrap();
        assert!(state.pending.unwrap().permits_linear_commit(linear, 3));
        assert!(!state.pending.unwrap().permits_tree_path(linear, &[0, 1, 2]));
        assert_eq!(state.resolve_prefix(linear, 3), Some(131));
    }

    #[test]
    fn compacted_path_advances_by_count_and_selects_physical_leaf_hidden() {
        let mut state = Gemma4DenseVerifierState {
            lane: Gemma4DenseSequenceLane::OrderedVerifier,
            logical_len: 128,
            next_ticket: 1,
            pending: None,
        };
        let pending = pending_tree(&mut state);
        let path = [0, 1, 5, 6, 7];
        assert!(pending.permits_tree_path(pending.ticket, &path));
        let hidden = (0..8).map(|i| vec![i as f32; 4]).collect::<Vec<_>>();
        assert_eq!(hidden[*path.last().unwrap()], [7.0; 4]);
        // Runtime invokes this resolution only after successful GPU compaction.
        assert_eq!(state.resolve_prefix(pending.ticket, path.len()), Some(133));
        assert_eq!(state.logical_len, 133);
        assert!(state.pending.is_none());
    }

    #[test]
    fn no_branch_acceptance_matches_linear_for_every_rejection_and_stop() {
        let tokens = [10, 11, 12, 13, 14, 15, 16, 17];
        let parents = [-1, 0, 1, 2, 3, 4, 5, 6];
        let depths = [0, 1, 2, 3, 4, 5, 6, 7];
        for mismatch_at in 0..=8 {
            for stop_at in 0..=8 {
                let mut target = [11, 12, 13, 14, 15, 16, 17, 18];
                if mismatch_at < 8 {
                    target[mismatch_at] = 98;
                }
                if stop_at < 8 {
                    target[stop_at] = 99;
                }
                let tree =
                    mtp12_tree_policy::accept(&tokens, &parents, &depths, &target, &[99], 100)
                        .unwrap();
                let linear =
                    gemma4_mtp12_acceptance_decision(&tokens[1..], &target, &[99]).unwrap();
                assert_eq!(
                    tree.path,
                    (0..linear.committed_input_rows).collect::<Vec<_>>()
                );
                assert_eq!(tree.stop_token, linear.stop_token);
                assert_eq!(tree.next_anchor_token, linear.next_anchor_token);
                assert_eq!(tree.path.len() - 1, linear.accepted_drafts);
            }
        }
    }
}

impl Gemma4PendingVerifierBatch {
    pub(super) fn permits_linear_commit(&self, ticket: u64, rows: usize) -> bool {
        self.ticket == ticket && rows <= self.width && (self.tree_parents.is_none() || rows == 0)
    }

    pub(super) fn permits_tree_path(&self, ticket: u64, path: &[usize]) -> bool {
        let Some(parents) = self.tree_parents else {
            return false;
        };
        self.ticket == ticket
            && self.width == 8
            && path.first() == Some(&0)
            && path.len() <= 8
            && path.iter().all(|&row| row < 8)
            && path
                .windows(2)
                .all(|pair| parents[pair[1]] == pair[0] as i32)
    }
}

impl Gemma4GpuRuntime {
    /// Execute the same eight projection columns with tree-specific semantic
    /// positions and ancestor attention. The completed ticket binds the tree.
    pub(crate) fn verify_tree_greedy(
        &self,
        candidate_tokens: &[u32],
        parents: &[i32],
        depths: &[u32],
        start_position: usize,
    ) -> Result<Gemma4DenseVerifierBatch> {
        self.verify_tree_greedy_with_glue(candidate_tokens, parents, depths, start_position, None)
    }

    /// [`Self::verify_tree_greedy`] with an explicit verifier fused-glue mask
    /// (`None` = `CAMELID_GEMMA4_VERIFY_FUSED_GLUE`; `Some(mask)` pins this
    /// call so one process can A/B the fused and legacy decoder encodes).
    pub(crate) fn verify_tree_greedy_with_glue(
        &self,
        candidate_tokens: &[u32],
        parents: &[i32],
        depths: &[u32],
        start_position: usize,
        fused_glue: Option<u32>,
    ) -> Result<Gemma4DenseVerifierBatch> {
        if !mtp12_tree_policy::validate(candidate_tokens, parents, depths, self.vocab)
            || !self.head_on_cpu
            || start_position
                .checked_add(8)
                .is_none_or(|end| end > self.model.max_positions())
        {
            return Err(BackendError::RuntimeShapeMismatch(
                "invalid bounded W8 target tree, vocabulary, head or physical capacity".into(),
            ));
        }
        let plan = crate::metal::Gemma4DenseTreePlan::new(parents, depths).ok_or_else(|| {
            BackendError::RuntimeShapeMismatch("Metal refused the W8 tree topology".into())
        })?;
        let head = self.q6k_gpu_head.as_ref().ok_or_else(|| {
            BackendError::UnsupportedModelArchitecture(
                "tree verifier requires SPEC50 Q6_K head".into(),
            )
        })?;
        let mut state = self.verifier_state.lock().map_err(|_| {
            BackendError::UnsupportedModelArchitecture("verifier state poisoned".into())
        })?;
        if !state.claim_ordered_verifier(start_position)
            || state.pending.is_some()
            || state.logical_len != start_position
        {
            return Err(BackendError::RuntimeShapeMismatch(
                "tree verifier requires the current committed ordered-Q4 cursor".into(),
            ));
        }
        let mut h0_rows = Vec::with_capacity(8 * self.hidden);
        let mut inputs_by_row = Vec::with_capacity(8);
        for (&token, &depth) in candidate_tokens.iter().zip(depths) {
            let (h0, inputs) =
                self.dense_verifier_row_inputs(token, start_position + depth as usize)?;
            h0_rows.extend_from_slice(&h0);
            inputs_by_row.push(inputs);
        }
        let hidden_flat = self
            .model
            .verify_tree_hidden_ordered_q4_with_glue(
                &h0_rows,
                &inputs_by_row,
                start_position,
                &plan,
                fused_glue,
            )
            .ok_or_else(|| {
                BackendError::UnsupportedModelArchitecture(
                    "ordered-Q4 tree decoder dispatch failed".into(),
                )
            })?;
        if hidden_flat.len() != 8 * self.hidden {
            return Err(BackendError::RuntimeShapeMismatch(
                "tree decoder returned invalid hidden count".into(),
            ));
        }
        let decoder_gpu_us =
            crate::metal::GEMMA4_LAST_VERIFY_GPU_US.load(std::sync::atomic::Ordering::Relaxed);
        let greedy_ids = head
            .forward_argmax_spec50_batch(&hidden_flat)
            .ok_or_else(|| {
                BackendError::UnsupportedModelArchitecture("tree SPEC50 target head failed".into())
            })?;
        if greedy_ids.len() != 8 {
            return Err(BackendError::RuntimeShapeMismatch(
                "tree head returned invalid ID count".into(),
            ));
        }
        let head_timing = head.last_spec50_timing();
        let ticket = state
            .record_completed_batch(start_position, 8)
            .ok_or_else(|| {
                BackendError::RuntimeShapeMismatch(
                    "tree verifier cursor changed during dispatch".into(),
                )
            })?;
        state
            .pending
            .as_mut()
            .expect("newly recorded tree ticket")
            .tree_parents = Some(parents.try_into().expect("validated eight parents"));
        Ok(Gemma4DenseVerifierBatch {
            ticket,
            start_position,
            greedy_ids,
            final_hidden: hidden_flat
                .chunks_exact(self.hidden)
                .map(<[f32]>::to_vec)
                .collect(),
            head_timing,
            decoder_gpu_us,
        })
    }

    /// Compact only a root-to-node path bound to this completed ticket. The
    /// committed prefix remains invisible until the GPU copy has completed.
    pub(crate) fn commit_verifier_tree_path(&self, ticket: u64, path: &[usize]) -> Result<usize> {
        let mut state = self.verifier_state.lock().map_err(|_| {
            BackendError::UnsupportedModelArchitecture("verifier state poisoned".into())
        })?;
        let pending = state.pending.ok_or_else(|| {
            BackendError::RuntimeShapeMismatch("no pending tree verifier batch".into())
        })?;
        let parents = pending.tree_parents.ok_or_else(|| {
            BackendError::RuntimeShapeMismatch("tree commit cannot consume a linear ticket".into())
        })?;
        if !pending.permits_tree_path(ticket, path) {
            return Err(BackendError::RuntimeShapeMismatch(
                "invalid tree commit ticket or ancestor path".into(),
            ));
        }
        let mut depths = [0; 8];
        for i in 1..8 {
            depths[i] = depths[parents[i] as usize] + 1;
        }
        let plan = crate::metal::Gemma4DenseTreePlan::new(&parents, &depths)
            .expect("completed ticket contains a validated tree");
        if self
            .model
            .compact_tree_kv_path(pending.start_position, &plan, path)
            .is_none()
        {
            // Compaction touches only tentative slots >= start_position. Even
            // a partial failed copy leaves the old committed prefix valid.
            let _ = state.resolve_prefix(ticket, 0);
            return Err(BackendError::UnsupportedModelArchitecture(
                "tree KV compaction failed; tentative batch was aborted".into(),
            ));
        }
        state.resolve_prefix(ticket, path.len()).ok_or_else(|| {
            BackendError::RuntimeShapeMismatch(
                "tree cursor resolution failed after compaction".into(),
            )
        })
    }

    pub(super) fn propose_mtp12_tree_ordered_q4(
        &self,
        assistant: &mut crate::metal::Gemma4Mtp12AssistantMetal,
        anchor_token: u32,
        pending_target_raw_hidden: &[f32],
        position: usize,
    ) -> Result<crate::metal::Gemma4Mtp12TreeProposal> {
        self.require_mtp12_target_identity()?;
        if pending_target_raw_hidden.len() != self.hidden
            || pending_target_raw_hidden
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 MTP raw target hidden has width {} (expected {}) or non-finite values",
                pending_target_raw_hidden.len(),
                self.hidden,
            )));
        }
        let normalized_hidden =
            rms_norm(pending_target_raw_hidden, Some(&self.output_norm), self.eps);
        if normalized_hidden.iter().any(|value| !value.is_finite()) {
            return Err(BackendError::RuntimeShapeMismatch(
                "Gemma 4 MTP normalized target hidden contains non-finite values".into(),
            ));
        }

        let sequence = self.verifier_state.lock().map_err(|_| {
            BackendError::UnsupportedModelArchitecture("verifier state poisoned".into())
        })?;
        if sequence.lane != Gemma4DenseSequenceLane::OrderedVerifier
            || sequence.pending.is_some()
            || sequence.logical_len != position
        {
            return Err(BackendError::UnsupportedModelArchitecture(format!(
                "MTP assistant requires an ordered committed target prefix: lane={:?}, logical_len={}, requested_position={position}, pending={}",
                sequence.lane,
                sequence.logical_len,
                sequence.pending.is_some(),
            )));
        }

        let head = self.q6k_gpu_head.as_ref().ok_or_else(|| {
            BackendError::UnsupportedModelArchitecture(
                "MTP assistant requires the strict mapped Q6_K target head".into(),
            )
        })?;
        let kv_scoped = head
            .with_full_table_device(
                crate::metal::GEMMA4_12B_QAT_Q4_0_TARGET_SHA256,
                |target_table| {
                    self.model.with_kv_device_views(
                        &[
                            crate::metal::GEMMA4_12B_MTP_SLIDING_HOST_LAYER,
                            crate::metal::GEMMA4_12B_MTP_FULL_HOST_LAYER,
                        ],
                        |views| {
                            if views.len() != 2
                                || views[0].source_layer
                                    != crate::metal::GEMMA4_12B_MTP_SLIDING_HOST_LAYER
                                || views[1].source_layer
                                    != crate::metal::GEMMA4_12B_MTP_FULL_HOST_LAYER
                            {
                                return Err(BackendError::UnsupportedModelArchitecture(
                                    "MTP target KV callback did not return exact source layers 46/47"
                                        .into(),
                                ));
                            }
                            let sliding_view = &views[0];
                            let sliding = crate::metal::Gemma4Mtp12DeviceKv {
                                key: crate::metal::Gemma4Mtp12MetalBufferView {
                                    buffer: sliding_view.key,
                                    byte_offset: sliding_view.byte_offset,
                                    byte_len: sliding_view.byte_len,
                                },
                                value: crate::metal::Gemma4Mtp12MetalBufferView {
                                    buffer: sliding_view.value,
                                    byte_offset: sliding_view.byte_offset,
                                    byte_len: sliding_view.byte_len,
                                },
                                source_layer: sliding_view.source_layer,
                                kv_heads: sliding_view.kv_heads,
                                head_dim: sliding_view.head_dim,
                                max_positions: sliding_view.max_positions,
                            };
                            let full_view = &views[1];
                            let full = crate::metal::Gemma4Mtp12DeviceKv {
                                key: crate::metal::Gemma4Mtp12MetalBufferView {
                                    buffer: full_view.key,
                                    byte_offset: full_view.byte_offset,
                                    byte_len: full_view.byte_len,
                                },
                                value: crate::metal::Gemma4Mtp12MetalBufferView {
                                    buffer: full_view.value,
                                    byte_offset: full_view.byte_offset,
                                    byte_len: full_view.byte_len,
                                },
                                source_layer: full_view.source_layer,
                                kv_heads: full_view.kv_heads,
                                head_dim: full_view.head_dim,
                                max_positions: full_view.max_positions,
                            };
                            assistant.propose_tree_w8_from_cpu_hidden(
                                anchor_token,
                                &normalized_hidden,
                                target_table,
                                sliding,
                                full,
                                position,
                                position,
                            )
                        },
                    )
                },
            )
            .ok_or_else(|| {
                BackendError::UnsupportedModelArchitecture(
                    "MTP target Q6_K full-table scope refused pinned identity or geometry".into(),
                )
            })?;
        let proposal = kv_scoped.ok_or_else(|| {
            BackendError::UnsupportedModelArchitecture(
                "MTP target could not expose resident KV source layers 46/47".into(),
            )
        })?;
        // Keep `sequence` alive through the complete synchronous assistant wait;
        // dropping it earlier would permit the target prefix to change while the
        // assistant is reading its borrowed KV buffers.
        drop(sequence);
        proposal
    }
}
