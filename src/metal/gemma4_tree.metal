#include <metal_stdlib>
using namespace metal;

struct Gemma4AttnV2Args {
    uint n_heads, head_dim, rows, group;
    float scale;
    uint position_stride, kv_head_stride, kv_base_offset;
    uint union_start, union_end, score_blocks, dim_blocks;
};

// Prefix scores use the original fused V2 shader. Only these <=8 logical suffix
// slots need node-specific K addressing. Retain the original scalar dot nest.
kernel void gemma4_tree_scores_suffix(
    device const float* query [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device float* scores [[buffer(3)]],
    constant Gemma4AttnV2Args& args [[buffer(5)]],
    constant uint4* row_meta [[buffer(6)]],
    constant uint* ancestors [[buffer(8)]],
    constant uint& tree_base [[buffer(9)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint head = tg.x;
    uint row = tg.y;
    if (head >= args.n_heads || row >= args.rows) return;
    uint4 meta = row_meta[row];
    uint position_count = meta.y;
    uint head_dim = args.head_dim;
    uint kv_head = head / args.group;
    uint q_base = row * args.n_heads * head_dim + head * head_dim;
    uint position_stride = args.position_stride;
    uint kv_base = args.kv_base_offset + kv_head * args.kv_head_stride;
    uint score_base = meta.z + head * position_count;
    float scale = args.scale;
    // One group handles this node's <=8 suffix positions. Committed-prefix
    // score groups already ran; no context-sized grid of empty work is needed.
    const uint suffix_count = meta.w - tree_base;
    for (uint suffix = lane; suffix < suffix_count; suffix += 32u) {
        uint logical_position = tree_base + suffix;
        if (logical_position < meta.x) continue;
        uint p = logical_position - meta.x;
        uint physical_position = tree_base + ancestors[row * 8u + suffix];
        uint k_base = kv_base + physical_position * position_stride;
        float s = 0.0;
        for (uint d = 0; d < head_dim; ++d) {
            s += query[q_base + d] * keys[k_base + d];
        }
        s *= scale;
        scores[score_base + p] = s;
    }
}

// Global-only candidate: preserve the qualified per-term FMA while keeping
// the long committed walk free of tree address selection. The HD256 control
// nest below remains unchanged; the selected HD256 path uses its own PAIRS2.
inline void gemma4_tree_context_hd512_split(
    device const float* values,
    device const float* scores,
    device float* output,
    constant Gemma4AttnV2Args& args,
    constant uint4* row_meta,
    device const float* denom_in,
    constant uint* ancestors,
    constant uint& tree_base,
    uint3 tg,
    uint lane)
{
    uint head = tg.x;
    uint row = tg.y;
    if (head >= args.n_heads || row >= args.rows) return;
    uint4 meta = row_meta[row];
    uint position_count = meta.y;
    constexpr uint head_dim = 512u;
    uint kv_head = head / args.group;
    uint q_base = row * args.n_heads * head_dim + head * head_dim;
    uint position_stride = args.position_stride;
    uint kv_base = args.kv_base_offset + kv_head * args.kv_head_stride;
    uint score_base = meta.z + head * position_count;
    float inv = 1.0 / denom_in[row * args.n_heads + head];
    uint stride = 32u * args.dim_blocks;
    const uint prefix_count = meta.x < tree_base
        ? min(position_count, tree_base - meta.x) : 0u;
    const uint prefix_kv_base = kv_base + meta.x * position_stride;

    for (uint d = lane + tg.z * 32u; d < head_dim; d += stride) {
        float acc = 0.0;
        for (uint p = 0; p < prefix_count; ++p) {
            const float v = values[prefix_kv_base + p * position_stride + d];
            acc = metal::fma(scores[score_base + p] * v, inv, acc);
        }
        for (uint p = prefix_count; p < position_count; ++p) {
            uint physical_position = tree_base + ancestors[row * 8u + meta.x + p - tree_base];
            const float v = values[kv_base + physical_position * position_stride + d];
            acc = metal::fma(scores[score_base + p] * v, inv, acc);
        }
        output[q_base + d] = acc;
    }
}

// The logical score positions, denominator, reciprocal location and ascending
// scalar fold are unchanged. Only each logical suffix V address is mapped.
kernel void gemma4_tree_context_nest(
    device const float* values [[buffer(2)]],
    device const float* scores [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant Gemma4AttnV2Args& args [[buffer(5)]],
    constant uint4* row_meta [[buffer(6)]],
    device const float* denom_in [[buffer(7)]],
    constant uint* ancestors [[buffer(8)]],
    constant uint& tree_base [[buffer(9)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (args.head_dim == 512u) {
        gemma4_tree_context_hd512_split(values, scores, output, args, row_meta,
            denom_in, ancestors, tree_base, tg, lane);
        return;
    }
    uint head = tg.x;
    uint row = tg.y;
    if (head >= args.n_heads || row >= args.rows) return;
    uint4 meta = row_meta[row];
    uint position_count = meta.y;
    uint head_dim = args.head_dim;
    uint kv_head = head / args.group;
    uint q_base = row * args.n_heads * head_dim + head * head_dim;
    uint position_stride = args.position_stride;
    uint kv_base = args.kv_base_offset + kv_head * args.kv_head_stride;
    uint score_base = meta.z + head * position_count;
    float inv = 1.0 / denom_in[row * args.n_heads + head];
    uint stride = 32u * args.dim_blocks;

    for (uint d = lane + tg.z * 32u; d < head_dim; d += stride) {
        float acc = 0.0;
        for (uint p = 0; p < position_count; ++p) {
            uint logical_position = meta.x + p;
            uint physical_position = logical_position < tree_base ? logical_position
                : tree_base + ancestors[row * 8u + logical_position - tree_base];
            acc += scores[score_base + p] * inv * values[kv_base + physical_position * position_stride + d];
        }
        output[q_base + d] = acc;
    }
}

// One threadgroup owns an entire (KV head, K-or-V) path. All input bits are
// staged before any destination is written, so overlapping paths such as
// [0,2,3] cannot overwrite an input another row still needs. Prefix <base is
// untouched. Maximum staging is8 rows*512 dims*4 bytes=16KiB per group.
kernel void gemma4_tree_compact_kv(
    device uint* keys [[buffer(0)]],
    device uint* values [[buffer(1)]],
    constant uint4& args [[buffer(2)]], // base, max_positions, head_dim, path_len
    constant uint* path [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uint staged[8u * 512u];
    device uint* data = tg.y == 0u ? keys : values;
    uint head_dim = args.z;
    uint head_base = tg.x * args.y * head_dim;
    uint elements = args.w * head_dim;
    for (uint i = tid; i < elements; i += 128u) {
        uint row = i / head_dim;
        uint d = i - row * head_dim;
        staged[i] = data[head_base + (args.x + path[row]) * head_dim + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < elements; i += 128u) {
        data[head_base + args.x * head_dim + i] = staged[i];
    }
}

// Sliding-context PAIRS2 retains the selected linear V2 geometry. Each output
// still folds its packed logical positions in ascending order. Committed V is
// shared by both pairs; only the <=8 suffix positions use per-node V addresses.
kernel void gemma4_tree_context_hd256_p2(
    device const float* values [[buffer(2)]],
    device const float* scores [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant Gemma4AttnV2Args& args [[buffer(5)]],
    constant uint4* row_meta [[buffer(6)]],
    device const float* denom_in [[buffer(7)]],
    constant uint* ancestors [[buffer(8)]],
    constant uint& tree_base [[buffer(9)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    constexpr uint HEAD_DIM = 256u;
    constexpr uint PAIRS = 2u;
    const uint kv_head = tg.x;
    const uint rows = args.rows;
    const uint group = args.group;
    const uint total_pairs = group * rows;
    const uint pair_begin = tg.y * PAIRS;
    if (pair_begin >= total_pairs) return;
    const uint pair_count = min(PAIRS, total_pairs - pair_begin);
    const uint d0 = tg.z * 32u + lane;
    if (d0 >= HEAD_DIM) return;
    const uint row_stride = args.n_heads * HEAD_DIM;

    uint ws[PAIRS];
    uint cnt[PAIRS];
    uint sbase[PAIRS];
    uint node[PAIRS];
    float inv[PAIRS];
    float acc[PAIRS];
#pragma clang loop unroll(full)
    for (uint j = 0u; j < PAIRS; ++j) {
        acc[j] = 0.0f;
        const uint pair = pair_begin + (j < pair_count ? j : 0u);
        const uint head_local = pair / rows;
        const uint row = pair - head_local * rows;
        const uint head = kv_head * group + head_local;
        const uint4 meta = row_meta[row];
        ws[j] = meta.x;
        cnt[j] = meta.y;
        sbase[j] = meta.z + head * meta.y;
        node[j] = row;
        inv[j] = 1.0 / denom_in[row * args.n_heads + head];
    }
    device const float* vbase =
        values + args.kv_base_offset + kv_head * args.kv_head_stride + d0;
    const uint position_stride = args.position_stride;

    // The prefix is the original PAIRS2 loop: one shared V load, no tree
    // address selection in the long walk. Every pair retains its same scalar
    // accumulator and reciprocal across the following <=8 suffix positions.
    // Exact fixture diagnostics identify the original compiler form as
    // score*value followed by inverse/acc FMA; pin that form in both loops.
    const uint prefix_end = min(args.union_end, tree_base);
    for (uint p = args.union_start; p < prefix_end; ++p) {
        const float v = vbase[p * position_stride];
#pragma clang loop unroll(full)
        for (uint j = 0u; j < PAIRS; ++j) {
            const uint rel = p - ws[j];
            if (p >= ws[j] && rel < cnt[j]) {
                acc[j] = metal::fma(scores[sbase[j] + rel] * v, inv[j], acc[j]);
            }
        }
    }
    for (uint p = max(args.union_start, tree_base); p < args.union_end; ++p) {
#pragma clang loop unroll(full)
        for (uint j = 0u; j < PAIRS; ++j) {
            const uint rel = p - ws[j];
            if (p >= ws[j] && rel < cnt[j]) {
                const uint physical = tree_base + ancestors[node[j] * 8u + p - tree_base];
                const float v = vbase[physical * position_stride];
                acc[j] = metal::fma(scores[sbase[j] + rel] * v, inv[j], acc[j]);
            }
        }
    }

#pragma clang loop unroll(full)
    for (uint j = 0u; j < PAIRS; ++j) {
        if (j < pair_count) {
            const uint pair = pair_begin + j;
            const uint head_local = pair / rows;
            const uint row = pair - head_local * rows;
            const uint head = kv_head * group + head_local;
            output[row * row_stride + head * HEAD_DIM + d0] = acc[j];
        }
    }
}

// Pipelined, V-sharing, branch-free context for the committed prefix (opt-in via
// CAMELID_GEMMA4_TREE_CONTEXT_FORM; the p2/nest kernels above stay the default).
// Per output element the fold is still ONE lane's strictly ascending logical
// positions from acc = 0.0f with the pinned statement
//     acc = metal::fma(score * v, inv, acc),   inv = 1.0 / denom  (once per pair)
// and the same lane->dim assignment; only the schedule changes:
//  (1) HEAD     [union_start, head_end): the p2 predicated loop verbatim. These are
//      the <= window-1 positions where some pair is not yet inside its window.
//  (2) INTERIOR [head_end, prefix_end) in 32-position blocks with every pair
//      active: scores loaded coalesced once per block (lane l holds position p+l)
//      and shared with simd_broadcast (a bit move), V loads software
//      double-buffered one 8-position sub-block ahead of the FMAs.
//  (3) TAIL     the < 32 remaining prefix positions, unconditional scalar form.
//  (4) SUFFIX   the p2 mapped loop verbatim (per-node V addresses via ancestors).
// Why the predicate may be dropped in (2)/(3): head_end = min(prefix_end, max_j ws[j]).
// Whenever those loops run, p >= head_end == max_j ws[j] >= ws[j] for every pair
// (if max_j ws[j] > prefix_end the trip counts are zero and HEAD folded everything),
// and p < prefix_end <= tree_base < tree_base + depth[j] + 1 == ws[j] + cnt[j], so
// `p >= ws[j] && p - ws[j] < cnt[j]` holds for every folded position; in the
// 32-block form p + 31 < prefix_end <= tree_base also keeps every coalesced score
// load inside the pair's packed row. Dropping the predicate therefore removes no
// term and adds none. No -inf masking, no vector math, no reassociation: the
// product is a separately rounded IEEE multiply of the same two operands and the
// explicit fma cannot be re-contracted. The interior touches only soff/inv/acc/sreg
// (ws/cnt/node stay in HEAD/SUFFIX) to keep register pressure low.
template <uint HEAD_DIM, uint PAIRS>
inline void gemma4_tree_context_pipelined(
    device const float* values,
    device const float* scores,
    device float* output,
    constant Gemma4AttnV2Args& args,
    constant uint4* row_meta,
    device const float* denom_in,
    constant uint* ancestors,
    constant uint& tree_base,
    uint3 tg,
    uint lane)
{
    const uint kv_head = tg.x;
    const uint rows = args.rows;
    const uint group = args.group;
    const uint total_pairs = group * rows;
    const uint pair_begin = tg.y * PAIRS;
    if (pair_begin >= total_pairs) return;
    const uint pair_count = min(PAIRS, total_pairs - pair_begin);
    // HEAD_DIM % 32 == 0 and the grid's depth is HEAD_DIM / 32, so every lane of
    // every simdgroup is active here (simd_broadcast below needs all 32 lanes).
    const uint d0 = tg.z * 32u + lane;
    if (d0 >= HEAD_DIM) return;
    const uint row_stride = args.n_heads * HEAD_DIM;

    uint ws[PAIRS];
    uint cnt[PAIRS];
    uint sbase[PAIRS];
    uint node[PAIRS];
    uint soff[PAIRS];
    float inv[PAIRS];
    float acc[PAIRS];
    uint ws_max = 0u;
#pragma clang loop unroll(full)
    for (uint j = 0u; j < PAIRS; ++j) {
        acc[j] = 0.0f;
        const uint pair = pair_begin + (j < pair_count ? j : 0u);
        const uint head_local = pair / rows;
        const uint row = pair - head_local * rows;
        const uint head = kv_head * group + head_local;
        const uint4 meta = row_meta[row];
        ws[j] = meta.x;
        cnt[j] = meta.y;
        sbase[j] = meta.z + head * meta.y;
        node[j] = row;
        inv[j] = 1.0 / denom_in[row * args.n_heads + head];
        // scores[sbase[j] + (p - ws[j])] == scores[soff[j] + p] (modular uint).
        soff[j] = sbase[j] - ws[j];
        ws_max = max(ws_max, ws[j]);
    }
    device const float* vbase =
        values + args.kv_base_offset + kv_head * args.kv_head_stride + d0;
    const uint position_stride = args.position_stride;

    const uint prefix_end = min(args.union_end, tree_base);
    const uint head_end = min(prefix_end, ws_max);
    uint p = args.union_start;

    // (1) HEAD: the p2 prefix loop verbatim, only its upper bound is head_end.
    for (; p < head_end; ++p) {
        const float v = vbase[p * position_stride];
#pragma clang loop unroll(full)
        for (uint j = 0u; j < PAIRS; ++j) {
            const uint rel = p - ws[j];
            if (p >= ws[j] && rel < cnt[j]) {
                acc[j] = metal::fma(scores[sbase[j] + rel] * v, inv[j], acc[j]);
            }
        }
    }

    // (2) INTERIOR: 32-position blocks, every pair active (see the proof above).
    float vbuf[8];
    if (p + 32u <= prefix_end) {
#pragma clang loop unroll(full)
        for (uint t = 0u; t < 8u; ++t) {
            vbuf[t] = vbase[(p + t) * position_stride];
        }
    }
    for (; p + 32u <= prefix_end; p += 32u) {
        float sreg[PAIRS];
#pragma clang loop unroll(full)
        for (uint j = 0u; j < PAIRS; ++j) {
            sreg[j] = scores[soff[j] + p + lane];
        }
#pragma clang loop unroll(full)
        for (uint sub = 0u; sub < 4u; ++sub) {
            float v[8];
#pragma clang loop unroll(full)
            for (uint t = 0u; t < 8u; ++t) {
                v[t] = vbuf[t];
            }
            // Issue the next sub-block's V loads (after sub 3: the next block's
            // first sub-block, if any) before this sub-block's FMAs.
            const uint next = p + (sub + 1u) * 8u;
            if (sub + 1u < 4u || next + 32u <= prefix_end) {
#pragma clang loop unroll(full)
                for (uint t = 0u; t < 8u; ++t) {
                    vbuf[t] = vbase[(next + t) * position_stride];
                }
            }
#pragma clang loop unroll(full)
            for (uint t = 0u; t < 8u; ++t) {
#pragma clang loop unroll(full)
                for (uint j = 0u; j < PAIRS; ++j) {
                    acc[j] = metal::fma(
                        simd_broadcast(sreg[j], (ushort)(sub * 8u + t)) * v[t], inv[j], acc[j]);
                }
            }
        }
    }

    // (3) TAIL: fewer than 32 prefix positions remain; every pair is active.
    for (; p < prefix_end; ++p) {
        const float v = vbase[p * position_stride];
#pragma clang loop unroll(full)
        for (uint j = 0u; j < PAIRS; ++j) {
            acc[j] = metal::fma(scores[soff[j] + p] * v, inv[j], acc[j]);
        }
    }

    // (4) SUFFIX: the p2 mapped loop verbatim.
    for (uint q = max(args.union_start, tree_base); q < args.union_end; ++q) {
#pragma clang loop unroll(full)
        for (uint j = 0u; j < PAIRS; ++j) {
            const uint rel = q - ws[j];
            if (q >= ws[j] && rel < cnt[j]) {
                const uint physical = tree_base + ancestors[node[j] * 8u + q - tree_base];
                const float v = vbase[physical * position_stride];
                acc[j] = metal::fma(scores[sbase[j] + rel] * v, inv[j], acc[j]);
            }
        }
    }

#pragma clang loop unroll(full)
    for (uint j = 0u; j < PAIRS; ++j) {
        if (j < pair_count) {
            const uint pair = pair_begin + j;
            const uint head_local = pair / rows;
            const uint row = pair - head_local * rows;
            const uint head = kv_head * group + head_local;
            output[row * row_stride + head * HEAD_DIM + d0] = acc[j];
        }
    }
}

#define GEMMA4_TREE_CONTEXT_PIPELINED_KERNEL(NAME, HD, PAIRS) \
kernel void NAME( \
    device const float* values [[buffer(2)]], \
    device const float* scores [[buffer(3)]], \
    device float* output [[buffer(4)]], \
    constant Gemma4AttnV2Args& args [[buffer(5)]], \
    constant uint4* row_meta [[buffer(6)]], \
    device const float* denom_in [[buffer(7)]], \
    constant uint* ancestors [[buffer(8)]], \
    constant uint& tree_base [[buffer(9)]], \
    uint3 tg [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]]) \
{ \
    gemma4_tree_context_pipelined<HD, PAIRS>( \
        values, scores, output, args, row_meta, denom_in, ancestors, tree_base, tg, lane); \
}

// Grid (n_kv_heads, ceil(group*rows / PAIRS), HD/32) x 32 lanes. At K=8: HD256 P2 =
// 512 threadgroups, P4 = 256, P8 = 128; HD512 P2 = 1024, P4 = 512, P8 = 256, P16 = 128. Fewer
// pairs per threadgroup means more V reads but more resident simdgroups; at K=8 the
// kernel is occupancy-bound, so the small-PAIRS forms are the candidates and P16 is
// measurement-only. The selection is made by the receipt, never by default.
GEMMA4_TREE_CONTEXT_PIPELINED_KERNEL(gemma4_tree_context_hd256_p2x, 256u, 2u)
GEMMA4_TREE_CONTEXT_PIPELINED_KERNEL(gemma4_tree_context_hd256_p4x, 256u, 4u)
GEMMA4_TREE_CONTEXT_PIPELINED_KERNEL(gemma4_tree_context_hd256_p8x, 256u, 8u)
GEMMA4_TREE_CONTEXT_PIPELINED_KERNEL(gemma4_tree_context_hd512_p2x, 512u, 2u)
GEMMA4_TREE_CONTEXT_PIPELINED_KERNEL(gemma4_tree_context_hd512_p4x, 512u, 4u)
GEMMA4_TREE_CONTEXT_PIPELINED_KERNEL(gemma4_tree_context_hd512_p8x, 512u, 8u)
GEMMA4_TREE_CONTEXT_PIPELINED_KERNEL(gemma4_tree_context_hd512_p16x, 512u, 16u)
