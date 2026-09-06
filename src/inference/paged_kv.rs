//! Paged KV-cache block allocation, page table mapping, and copy-on-write management.
//!
//! Inspired by PagedAttention (vLLM / SGLang), this module provides a virtual memory
//! block allocator for K/V tensors. Instead of contiguous sequence allocation, tokens
//! are mapped to fixed-size physical blocks (`KV_BLOCK_TOKENS = 16`).
//!
//! Key properties:
//! - Zero memory fragmentation: Physical blocks are pooled and reused across sequences.
//! - Copy-on-Write (CoW): Shared prefix blocks (from system prompts or parallel branches)
//!   are reference-counted. A block is only cloned if mutated when `ref_count > 1`.
//! - Bit-exact precision: Direct dispatch to native F32, F16, Q8_0, Q4_0, FP8_E4M3, and FP8_E5M2 kernels.

use crate::inference::kv_cache::{KvDtype, LlamaKvCachePlan};
use crate::tensor::kv_quant::{
    axpy_row_fp8_e4m3, axpy_row_fp8_e5m2, axpy_row_q4_0, axpy_row_q8_0, quantize_row_fp8_e4m3,
    quantize_row_fp8_e5m2, quantize_row_q4_0, quantize_row_q8_0, vec_dot_row_fp8_e4m3,
    vec_dot_row_fp8_e5m2, vec_dot_row_q4_0, vec_dot_row_q8_0, BlockFp8E4m3, BlockFp8E5m2,
    BlockQ4_0, BlockQ8_0, KV_QUANT_BLOCK_VALUES,
};
use crate::{BackendError, Result};

/// Number of tokens stored per physical KV block.
pub const KV_BLOCK_TOKENS: usize = 16;

/// Unique physical block identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhysicalBlockId(pub u32);

impl PhysicalBlockId {
    #[inline]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// A physical storage block holding K/V activations for `KV_BLOCK_TOKENS` tokens.
#[derive(Debug, Clone)]
pub struct PhysicalKvBlock {
    pub id: PhysicalBlockId,
    pub ref_count: usize,
    pub token_ids: [u32; KV_BLOCK_TOKENS],
    pub num_tokens: usize,
    pub dtype: KvDtype,
    // Physical storage buffers
    pub keys_f32: Vec<f32>,
    pub values_f32: Vec<f32>,
    pub keys_f16: Vec<u16>,
    pub values_f16: Vec<u16>,
    pub keys_q8_0: Vec<BlockQ8_0>,
    pub values_q8_0: Vec<BlockQ8_0>,
    pub keys_q4_0: Vec<BlockQ4_0>,
    pub values_q4_0: Vec<BlockQ4_0>,
    pub keys_fp8_e4m3: Vec<BlockFp8E4m3>,
    pub values_fp8_e4m3: Vec<BlockFp8E4m3>,
    pub keys_fp8_e5m2: Vec<BlockFp8E5m2>,
    pub values_fp8_e5m2: Vec<BlockFp8E5m2>,
}

impl PhysicalKvBlock {
    pub fn new(id: PhysicalBlockId, plan: &LlamaKvCachePlan, dtype: KvDtype) -> Self {
        let streams = plan.layer_count * plan.kv_head_count;
        let k_dim = plan.k_head_dim;
        let v_dim = plan.v_head_dim;

        let mut block = Self {
            id,
            ref_count: 1,
            token_ids: [0; KV_BLOCK_TOKENS],
            num_tokens: 0,
            dtype,
            keys_f32: Vec::new(),
            values_f32: Vec::new(),
            keys_f16: Vec::new(),
            values_f16: Vec::new(),
            keys_q8_0: Vec::new(),
            values_q8_0: Vec::new(),
            keys_q4_0: Vec::new(),
            values_q4_0: Vec::new(),
            keys_fp8_e4m3: Vec::new(),
            values_fp8_e4m3: Vec::new(),
            keys_fp8_e5m2: Vec::new(),
            values_fp8_e5m2: Vec::new(),
        };

        match dtype {
            KvDtype::F32 => {
                block
                    .keys_f32
                    .resize(KV_BLOCK_TOKENS * streams * k_dim, 0.0);
                block
                    .values_f32
                    .resize(KV_BLOCK_TOKENS * streams * v_dim, 0.0);
            }
            KvDtype::F16 => {
                block.keys_f16.resize(KV_BLOCK_TOKENS * streams * k_dim, 0);
                block
                    .values_f16
                    .resize(KV_BLOCK_TOKENS * streams * v_dim, 0);
            }
            KvDtype::Q8_0 => {
                let k_blocks = k_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                let v_blocks = v_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                block
                    .keys_q8_0
                    .resize(KV_BLOCK_TOKENS * streams * k_blocks, BlockQ8_0::default());
                block
                    .values_q8_0
                    .resize(KV_BLOCK_TOKENS * streams * v_blocks, BlockQ8_0::default());
            }
            KvDtype::Q4_0 => {
                let k_blocks = k_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                let v_blocks = v_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                block
                    .keys_q4_0
                    .resize(KV_BLOCK_TOKENS * streams * k_blocks, BlockQ4_0::default());
                block
                    .values_q4_0
                    .resize(KV_BLOCK_TOKENS * streams * v_blocks, BlockQ4_0::default());
            }
            KvDtype::Fp8E4m3 => {
                let k_blocks = k_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                let v_blocks = v_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                block.keys_fp8_e4m3.resize(
                    KV_BLOCK_TOKENS * streams * k_blocks,
                    BlockFp8E4m3::default(),
                );
                block.values_fp8_e4m3.resize(
                    KV_BLOCK_TOKENS * streams * v_blocks,
                    BlockFp8E4m3::default(),
                );
            }
            KvDtype::Fp8E5m2 => {
                let k_blocks = k_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                let v_blocks = v_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                block.keys_fp8_e5m2.resize(
                    KV_BLOCK_TOKENS * streams * k_blocks,
                    BlockFp8E5m2::default(),
                );
                block.values_fp8_e5m2.resize(
                    KV_BLOCK_TOKENS * streams * v_blocks,
                    BlockFp8E5m2::default(),
                );
            }
        }
        block
    }

    #[inline]
    fn stream_idx(layer_idx: usize, kv_head: usize, plan: &LlamaKvCachePlan) -> usize {
        layer_idx * plan.kv_head_count + kv_head
    }

    pub fn store_kv(
        &mut self,
        token_offset: usize,
        layer_idx: usize,
        kv_head: usize,
        key: &[f32],
        value: &[f32],
        plan: &LlamaKvCachePlan,
    ) {
        assert!(token_offset < KV_BLOCK_TOKENS);
        let s_idx = Self::stream_idx(layer_idx, kv_head, plan);
        let k_dim = plan.k_head_dim;
        let v_dim = plan.v_head_dim;

        match self.dtype {
            KvDtype::F32 => {
                let k_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * k_dim;
                self.keys_f32[k_start..k_start + k_dim].copy_from_slice(key);
                if v_dim > 0 {
                    let v_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * v_dim;
                    self.values_f32[v_start..v_start + v_dim].copy_from_slice(value);
                }
            }
            KvDtype::F16 => {
                let k_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * k_dim;
                for (dst, &src) in self.keys_f16[k_start..k_start + k_dim].iter_mut().zip(key) {
                    *dst = crate::tensor::f32_to_f16_bits(src);
                }
                if v_dim > 0 {
                    let v_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * v_dim;
                    for (dst, &src) in self.values_f16[v_start..v_start + v_dim]
                        .iter_mut()
                        .zip(value)
                    {
                        *dst = crate::tensor::f32_to_f16_bits(src);
                    }
                }
            }
            KvDtype::Q8_0 => {
                let k_blocks = k_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                let k_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * k_blocks;
                quantize_row_q8_0(key, &mut self.keys_q8_0[k_start..k_start + k_blocks]);
                if v_dim > 0 {
                    let v_blocks = v_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                    let v_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * v_blocks;
                    quantize_row_q8_0(value, &mut self.values_q8_0[v_start..v_start + v_blocks]);
                }
            }
            KvDtype::Q4_0 => {
                let k_blocks = k_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                let k_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * k_blocks;
                quantize_row_q4_0(key, &mut self.keys_q4_0[k_start..k_start + k_blocks]);
                if v_dim > 0 {
                    let v_blocks = v_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                    let v_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * v_blocks;
                    quantize_row_q4_0(value, &mut self.values_q4_0[v_start..v_start + v_blocks]);
                }
            }
            KvDtype::Fp8E4m3 => {
                let k_blocks = k_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                let k_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * k_blocks;
                quantize_row_fp8_e4m3(key, &mut self.keys_fp8_e4m3[k_start..k_start + k_blocks]);
                if v_dim > 0 {
                    let v_blocks = v_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                    let v_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * v_blocks;
                    quantize_row_fp8_e4m3(
                        value,
                        &mut self.values_fp8_e4m3[v_start..v_start + v_blocks],
                    );
                }
            }
            KvDtype::Fp8E5m2 => {
                let k_blocks = k_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                let k_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * k_blocks;
                quantize_row_fp8_e5m2(key, &mut self.keys_fp8_e5m2[k_start..k_start + k_blocks]);
                if v_dim > 0 {
                    let v_blocks = v_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                    let v_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * v_blocks;
                    quantize_row_fp8_e5m2(
                        value,
                        &mut self.values_fp8_e5m2[v_start..v_start + v_blocks],
                    );
                }
            }
        }
    }

    pub fn dot_key(
        &self,
        token_offset: usize,
        layer_idx: usize,
        kv_head: usize,
        query: &[f32],
        plan: &LlamaKvCachePlan,
    ) -> f32 {
        let s_idx = Self::stream_idx(layer_idx, kv_head, plan);
        let k_dim = plan.k_head_dim;

        match self.dtype {
            KvDtype::F32 => {
                let k_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * k_dim;
                let k_slice = &self.keys_f32[k_start..k_start + k_dim];
                crate::tensor::dot_product(query, k_slice)
            }
            KvDtype::F16 => {
                let k_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * k_dim;
                let k_slice = &self.keys_f16[k_start..k_start + k_dim];
                query
                    .iter()
                    .zip(k_slice)
                    .map(|(&q, &k)| q * crate::tensor::f16_bits_to_f32(k))
                    .sum()
            }
            KvDtype::Q8_0 => {
                let k_blocks = k_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                let k_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * k_blocks;
                vec_dot_row_q8_0(query, &self.keys_q8_0[k_start..k_start + k_blocks])
            }
            KvDtype::Q4_0 => {
                let k_blocks = k_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                let k_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * k_blocks;
                vec_dot_row_q4_0(query, &self.keys_q4_0[k_start..k_start + k_blocks])
            }
            KvDtype::Fp8E4m3 => {
                let k_blocks = k_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                let k_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * k_blocks;
                vec_dot_row_fp8_e4m3(query, &self.keys_fp8_e4m3[k_start..k_start + k_blocks])
            }
            KvDtype::Fp8E5m2 => {
                let k_blocks = k_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                let k_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * k_blocks;
                vec_dot_row_fp8_e5m2(query, &self.keys_fp8_e5m2[k_start..k_start + k_blocks])
            }
        }
    }

    pub fn axpy_value(
        &self,
        token_offset: usize,
        layer_idx: usize,
        kv_head: usize,
        prob: f32,
        out: &mut [f32],
        plan: &LlamaKvCachePlan,
    ) {
        let s_idx = Self::stream_idx(layer_idx, kv_head, plan);
        let v_dim = plan.v_head_dim;
        if v_dim == 0 {
            return;
        }

        match self.dtype {
            KvDtype::F32 => {
                let v_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * v_dim;
                let v_slice = &self.values_f32[v_start..v_start + v_dim];
                for (o, &v) in out.iter_mut().zip(v_slice) {
                    *o += prob * v;
                }
            }
            KvDtype::F16 => {
                let v_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * v_dim;
                let v_slice = &self.values_f16[v_start..v_start + v_dim];
                for (o, &v) in out.iter_mut().zip(v_slice) {
                    *o += prob * crate::tensor::f16_bits_to_f32(v);
                }
            }
            KvDtype::Q8_0 => {
                let v_blocks = v_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                let v_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * v_blocks;
                axpy_row_q8_0(out, prob, &self.values_q8_0[v_start..v_start + v_blocks]);
            }
            KvDtype::Q4_0 => {
                let v_blocks = v_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                let v_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * v_blocks;
                axpy_row_q4_0(out, prob, &self.values_q4_0[v_start..v_start + v_blocks]);
            }
            KvDtype::Fp8E4m3 => {
                let v_blocks = v_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                let v_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * v_blocks;
                axpy_row_fp8_e4m3(
                    out,
                    prob,
                    &self.values_fp8_e4m3[v_start..v_start + v_blocks],
                );
            }
            KvDtype::Fp8E5m2 => {
                let v_blocks = v_dim.div_ceil(KV_QUANT_BLOCK_VALUES);
                let v_start = (s_idx * KV_BLOCK_TOKENS + token_offset) * v_blocks;
                axpy_row_fp8_e5m2(
                    out,
                    prob,
                    &self.values_fp8_e5m2[v_start..v_start + v_blocks],
                );
            }
        }
    }
}

/// Physical memory pool managing reusable fixed-size KV blocks.
#[derive(Debug)]
pub struct PagedKvBlockPool {
    pub plan: LlamaKvCachePlan,
    pub dtype: KvDtype,
    blocks: Vec<Option<PhysicalKvBlock>>,
    free_list: Vec<PhysicalBlockId>,
    max_blocks: usize,
    allocated_count: usize,
}

impl PagedKvBlockPool {
    pub fn new(plan: LlamaKvCachePlan, dtype: KvDtype, max_blocks: usize) -> Self {
        Self {
            plan,
            dtype,
            blocks: Vec::new(),
            free_list: Vec::new(),
            max_blocks,
            allocated_count: 0,
        }
    }

    /// Allocate a block from the pool or recycle from free list.
    pub fn allocate(&mut self) -> Result<PhysicalBlockId> {
        if let Some(recycled_id) = self.free_list.pop() {
            let block = self.blocks[recycled_id.as_usize()]
                .as_mut()
                .ok_or_else(|| {
                    BackendError::RuntimeShapeMismatch("Corrupt free block in pool".to_string())
                })?;
            block.ref_count = 1;
            block.num_tokens = 0;
            self.allocated_count += 1;
            return Ok(recycled_id);
        }

        if self.blocks.len() >= self.max_blocks {
            return Err(BackendError::KvCacheBudgetExceeded {
                positions: self.blocks.len() * KV_BLOCK_TOKENS,
                needed_bytes: self.block_bytes() as u64,
                budget_bytes: (self.max_blocks * self.block_bytes()) as u64,
            });
        }

        let new_id = PhysicalBlockId(self.blocks.len() as u32);
        let block = PhysicalKvBlock::new(new_id, &self.plan, self.dtype);
        self.blocks.push(Some(block));
        self.allocated_count += 1;
        Ok(new_id)
    }

    /// Increment reference count for shared block (e.g. prompt prefix caching or sequence branching).
    pub fn retain(&mut self, id: PhysicalBlockId) {
        if let Some(Some(block)) = self.blocks.get_mut(id.as_usize()) {
            block.ref_count += 1;
        }
    }

    /// Decrement reference count. If 0, return block to free list.
    pub fn release(&mut self, id: PhysicalBlockId) {
        if let Some(Some(block)) = self.blocks.get_mut(id.as_usize()) {
            // A block already at zero is on the free list. Re-releasing it previously
            // pushed the id a second time, after which two independent `allocate()`
            // calls handed back the SAME block and silently aliased their KV. Treat a
            // release of an already-free block as a no-op instead.
            // Deliberately a silent no-op rather than a debug assertion: this is the
            // guard that keeps a caller-side double free from corrupting the pool, so
            // it has to hold in every build profile, including tests.
            if block.ref_count == 0 {
                return;
            }
            block.ref_count -= 1;
            if block.ref_count == 0 {
                self.free_list.push(id);
                self.allocated_count = self.allocated_count.saturating_sub(1);
            }
        }
    }

    /// Copy-on-Write: If block has `ref_count > 1`, clone its contents into a newly allocated block
    /// and decrement old block's ref_count.
    pub fn ensure_unique(&mut self, id: PhysicalBlockId) -> Result<PhysicalBlockId> {
        let needs_cow = self.block(id).is_some_and(|b| b.ref_count > 1);
        if !needs_cow {
            return Ok(id);
        }

        let cloned_block = self
            .block(id)
            .ok_or_else(|| {
                BackendError::RuntimeShapeMismatch("Block not found for CoW".to_string())
            })?
            .clone();

        let new_id = self.allocate()?;
        let target = self.block_mut(new_id).ok_or_else(|| {
            BackendError::RuntimeShapeMismatch("Allocated block missing".to_string())
        })?;

        target.token_ids = cloned_block.token_ids;
        target.num_tokens = cloned_block.num_tokens;
        target.keys_f32.copy_from_slice(&cloned_block.keys_f32);
        target.values_f32.copy_from_slice(&cloned_block.values_f32);
        target.keys_f16.copy_from_slice(&cloned_block.keys_f16);
        target.values_f16.copy_from_slice(&cloned_block.values_f16);
        target.keys_q8_0.copy_from_slice(&cloned_block.keys_q8_0);
        target
            .values_q8_0
            .copy_from_slice(&cloned_block.values_q8_0);
        target.keys_q4_0.copy_from_slice(&cloned_block.keys_q4_0);
        target
            .values_q4_0
            .copy_from_slice(&cloned_block.values_q4_0);
        target
            .keys_fp8_e4m3
            .copy_from_slice(&cloned_block.keys_fp8_e4m3);
        target
            .values_fp8_e4m3
            .copy_from_slice(&cloned_block.values_fp8_e4m3);
        target
            .keys_fp8_e5m2
            .copy_from_slice(&cloned_block.keys_fp8_e5m2);
        target
            .values_fp8_e5m2
            .copy_from_slice(&cloned_block.values_fp8_e5m2);

        self.release(id);
        Ok(new_id)
    }

    #[inline]
    pub fn block(&self, id: PhysicalBlockId) -> Option<&PhysicalKvBlock> {
        self.blocks.get(id.as_usize()).and_then(|b| b.as_ref())
    }

    #[inline]
    pub fn block_mut(&mut self, id: PhysicalBlockId) -> Option<&mut PhysicalKvBlock> {
        self.blocks.get_mut(id.as_usize()).and_then(|b| b.as_mut())
    }

    pub fn allocated_count(&self) -> usize {
        self.allocated_count
    }

    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    pub fn block_bytes(&self) -> usize {
        let streams = self.plan.layer_count * self.plan.kv_head_count;
        let k_dim = self.plan.k_head_dim;
        let v_dim = self.plan.v_head_dim;
        let k_bytes = match self.dtype {
            KvDtype::F32 => k_dim * 4,
            KvDtype::F16 => k_dim * 2,
            KvDtype::Q8_0 => {
                k_dim.div_ceil(KV_QUANT_BLOCK_VALUES) * std::mem::size_of::<BlockQ8_0>()
            }
            KvDtype::Q4_0 => {
                k_dim.div_ceil(KV_QUANT_BLOCK_VALUES) * std::mem::size_of::<BlockQ4_0>()
            }
            KvDtype::Fp8E4m3 => {
                k_dim.div_ceil(KV_QUANT_BLOCK_VALUES) * std::mem::size_of::<BlockFp8E4m3>()
            }
            KvDtype::Fp8E5m2 => {
                k_dim.div_ceil(KV_QUANT_BLOCK_VALUES) * std::mem::size_of::<BlockFp8E5m2>()
            }
        };
        let v_bytes = match self.dtype {
            KvDtype::F32 => v_dim * 4,
            KvDtype::F16 => v_dim * 2,
            KvDtype::Q8_0 => {
                v_dim.div_ceil(KV_QUANT_BLOCK_VALUES) * std::mem::size_of::<BlockQ8_0>()
            }
            KvDtype::Q4_0 => {
                v_dim.div_ceil(KV_QUANT_BLOCK_VALUES) * std::mem::size_of::<BlockQ4_0>()
            }
            KvDtype::Fp8E4m3 => {
                v_dim.div_ceil(KV_QUANT_BLOCK_VALUES) * std::mem::size_of::<BlockFp8E4m3>()
            }
            KvDtype::Fp8E5m2 => {
                v_dim.div_ceil(KV_QUANT_BLOCK_VALUES) * std::mem::size_of::<BlockFp8E5m2>()
            }
        };
        KV_BLOCK_TOKENS * streams * (k_bytes + v_bytes)
    }
}

/// Logical-to-physical block mapping table for a sequence.
#[derive(Debug, Clone, Default)]
pub struct BlockTable {
    pub blocks: Vec<PhysicalBlockId>,
    pub num_tokens: usize,
}

impl BlockTable {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            num_tokens: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.num_tokens
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_tokens == 0
    }

    /// Map logical sequence position `pos` to `(PhysicalBlockId, token_offset_in_block)`.
    #[inline]
    pub fn logical_to_physical(&self, pos: usize) -> Option<(PhysicalBlockId, usize)> {
        if pos >= self.num_tokens {
            return None;
        }
        let block_idx = pos / KV_BLOCK_TOKENS;
        let token_offset = pos % KV_BLOCK_TOKENS;
        self.blocks.get(block_idx).map(|&id| (id, token_offset))
    }

    /// Append a token slot, allocating a new physical block if the last block is full.
    pub fn append_token(
        &mut self,
        pool: &mut PagedKvBlockPool,
        token_id: u32,
    ) -> Result<(PhysicalBlockId, usize)> {
        let block_idx = self.num_tokens / KV_BLOCK_TOKENS;
        let token_offset = self.num_tokens % KV_BLOCK_TOKENS;

        if block_idx == self.blocks.len() {
            let new_block_id = pool.allocate()?;
            self.blocks.push(new_block_id);
        } else {
            // If writing to existing block, ensure CoW uniqueness
            let unique_id = pool.ensure_unique(self.blocks[block_idx])?;
            self.blocks[block_idx] = unique_id;
        }

        let block_id = self.blocks[block_idx];
        if let Some(block) = pool.block_mut(block_id) {
            block.token_ids[token_offset] = token_id;
            block.num_tokens = token_offset + 1;
        }
        self.num_tokens += 1;
        Ok((block_id, token_offset))
    }

    /// Fork sequence (O(1) Copy-on-Write branch): increments ref count on all physical blocks.
    pub fn fork(&self, pool: &mut PagedKvBlockPool) -> Self {
        for &id in &self.blocks {
            pool.retain(id);
        }
        Self {
            blocks: self.blocks.clone(),
            num_tokens: self.num_tokens,
        }
    }

    /// Release all blocks held by this table back to pool.
    pub fn release(&mut self, pool: &mut PagedKvBlockPool) {
        for id in self.blocks.drain(..) {
            pool.release(id);
        }
        self.num_tokens = 0;
    }
}

/// Compute Paged Attention across non-contiguous physical blocks.
pub fn paged_attention(
    query: &[f32],
    layer_idx: usize,
    kv_head: usize,
    scale: f32,
    block_table: &BlockTable,
    pool: &PagedKvBlockPool,
    out: &mut [f32],
) -> Result<()> {
    let num_tokens = block_table.len();
    if num_tokens == 0 {
        return Ok(());
    }

    let mut scores = Vec::with_capacity(num_tokens);
    for pos in 0..num_tokens {
        let (block_id, token_offset) = block_table.logical_to_physical(pos).ok_or_else(|| {
            BackendError::RuntimeShapeMismatch("Paged attention bounds error".to_string())
        })?;
        let block = pool.block(block_id).ok_or_else(|| {
            BackendError::RuntimeShapeMismatch("Physical block missing".to_string())
        })?;
        let score = block.dot_key(token_offset, layer_idx, kv_head, query, &pool.plan) * scale;
        scores.push(score);
    }

    // Softmax
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut score_sum = 0.0f32;
    for score in scores.iter_mut() {
        *score = (*score - max_score).exp();
        score_sum += *score;
    }
    if score_sum == 0.0 || !score_sum.is_finite() {
        return Err(BackendError::RuntimeShapeMismatch(
            "paged attention softmax produced invalid normalization sum".to_string(),
        ));
    }
    let inv_score_sum = 1.0 / score_sum;

    // Value accumulation
    out.fill(0.0);
    for (pos, &score) in scores.iter().enumerate() {
        let prob = score * inv_score_sum;
        let (block_id, token_offset) = block_table.logical_to_physical(pos).unwrap();
        let block = pool.block(block_id).unwrap();
        block.axpy_value(token_offset, layer_idx, kv_head, prob, out, &pool.plan);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_plan() -> LlamaKvCachePlan {
        LlamaKvCachePlan {
            max_sequence_length: 128,
            layer_count: 2,
            kv_head_count: 2,
            head_dim: 64,
            k_head_dim: 64,
            v_head_dim: 64,
            key_shape: vec![1, 2, 128, 64],
            value_shape: vec![1, 2, 128, 64],
        }
    }

    #[test]
    fn test_paged_block_pool_allocation_and_free() {
        let plan = dummy_plan();
        let mut pool = PagedKvBlockPool::new(plan, KvDtype::F32, 10);
        let b1 = pool.allocate().unwrap();
        let b2 = pool.allocate().unwrap();
        assert_ne!(b1, b2);
        assert_eq!(pool.allocated_count(), 2);

        pool.release(b1);
        assert_eq!(pool.allocated_count(), 1);
        assert_eq!(pool.free_count(), 1);

        let b3 = pool.allocate().unwrap();
        assert_eq!(b3, b1, "Recycled block should be reused");
    }

    #[test]
    fn test_block_table_cow_forking() {
        let plan = dummy_plan();
        let mut pool = PagedKvBlockPool::new(plan, KvDtype::F32, 10);
        let mut table1 = BlockTable::new();

        for i in 0..20 {
            table1.append_token(&mut pool, i as u32).unwrap();
        }
        assert_eq!(table1.blocks.len(), 2); // 20 tokens -> 2 blocks of 16

        // Fork table2
        let mut table2 = table1.fork(&mut pool);
        assert_eq!(table2.blocks, table1.blocks);
        assert_eq!(pool.block(table1.blocks[0]).unwrap().ref_count, 2);

        // Appending to table2 triggers CoW on the last block
        table2.append_token(&mut pool, 100).unwrap();
        assert_ne!(
            table1.blocks[1], table2.blocks[1],
            "Block 1 should be cloned on write"
        );
        assert_eq!(
            pool.block(table1.blocks[0]).unwrap().ref_count,
            2,
            "Block 0 remains shared"
        );
    }

    #[test]
    fn test_paged_attention_fp8_parity() {
        let plan = dummy_plan();
        let mut pool = PagedKvBlockPool::new(plan.clone(), KvDtype::Fp8E4m3, 10);
        let mut table = BlockTable::new();

        let key_data = vec![0.25f32; 64];
        let val_data = vec![0.50f32; 64];

        for i in 0..4 {
            let (block_id, offset) = table.append_token(&mut pool, i as u32).unwrap();
            let block = pool.block_mut(block_id).unwrap();
            block.store_kv(offset, 0, 0, &key_data, &val_data, &plan);
        }

        let query = vec![0.1f32; 64];
        let mut out = vec![0.0f32; 64];
        paged_attention(&query, 0, 0, 0.125, &table, &pool, &mut out).unwrap();

        for &val in &out {
            assert!((val - 0.50).abs() < 0.05, "Expected ~0.50, got {val}");
        }
    }

    /// Regression: `release` decremented with `saturating_sub` and then pushed to the
    /// free list whenever the count read zero. Releasing an already-free block left the
    /// count at zero, so the condition fired again and the id landed on the free list a
    /// second time — after which two independent `allocate()` calls handed back the
    /// SAME block and silently aliased their KV.
    #[test]
    fn releasing_an_already_free_block_cannot_alias_it() {
        let mut pool = PagedKvBlockPool::new(dummy_plan(), KvDtype::F32, 8);
        let b = pool.allocate().unwrap();

        pool.release(b);
        let free_after_first = pool.free_count();
        pool.release(b);
        assert_eq!(
            pool.free_count(),
            free_after_first,
            "a redundant release must not enqueue the block twice"
        );

        let x = pool.allocate().unwrap();
        let y = pool.allocate().unwrap();
        assert_ne!(x, y, "two allocations must never return the same block");
    }

    /// Copy-on-write hands the writer a private block and drops its share of the
    /// original, leaving the other holder as sole owner.
    #[test]
    fn copy_on_write_splits_ownership_exactly_once() {
        let mut pool = PagedKvBlockPool::new(dummy_plan(), KvDtype::F32, 8);
        let shared = pool.allocate().unwrap();
        pool.retain(shared); // two holders
        assert_eq!(pool.block(shared).unwrap().ref_count, 2);

        let private = pool.ensure_unique(shared).unwrap();
        assert_ne!(private, shared, "a shared block must be cloned on write");
        assert_eq!(
            pool.block(shared).unwrap().ref_count,
            1,
            "the writer's share of the original is dropped"
        );
        assert_eq!(pool.block(private).unwrap().ref_count, 1);

        // An unshared block is returned as-is rather than needlessly copied.
        assert_eq!(pool.ensure_unique(private).unwrap(), private);
    }

    /// The pool is a fixed budget: exhausting it reports an error rather than
    /// over-allocating, and recycling makes capacity available again.
    #[test]
    fn exhausting_the_pool_reports_a_budget_error_and_recovers() {
        let capacity = 4;
        let mut pool = PagedKvBlockPool::new(dummy_plan(), KvDtype::F32, capacity);
        let blocks: Vec<_> = (0..capacity).map(|_| pool.allocate().unwrap()).collect();
        assert!(pool.allocate().is_err(), "capacity must be a hard bound");

        pool.release(blocks[0]);
        assert!(pool.allocate().is_ok(), "a freed block is reusable");
    }
}
