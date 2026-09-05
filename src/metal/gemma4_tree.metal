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
    uint stride = 32u * args.score_blocks;
    float scale = args.scale;

    for (uint p = lane + tg.z * 32u; p < position_count; p += stride) {
        uint logical_position = meta.x + p;
        if (logical_position < tree_base) continue;
        uint physical_position = tree_base + ancestors[row * 8u + logical_position - tree_base];
        uint k_base = kv_base + physical_position * position_stride;
        float s = 0.0;
        for (uint d = 0; d < head_dim; ++d) {
            s += query[q_base + d] * keys[k_base + d];
        }
        s *= scale;
        scores[score_base + p] = s;
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

    for (uint p = args.union_start; p < args.union_end; ++p) {
        const bool committed = p < tree_base;
        const float prefix_v = committed ? vbase[p * position_stride] : 0.0f;
#pragma clang loop unroll(full)
        for (uint j = 0u; j < PAIRS; ++j) {
            const uint rel = p - ws[j];
            if (p >= ws[j] && rel < cnt[j]) {
                const uint physical = committed ? p
                    : tree_base + ancestors[node[j] * 8u + p - tree_base];
                const float v = committed ? prefix_v : vbase[physical * position_stride];
                acc[j] += scores[sbase[j] + rel] * inv[j] * v;
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
