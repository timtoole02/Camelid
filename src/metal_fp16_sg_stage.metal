// Linked after LINEAR_ROW_SHADER using its original compile options.
// Canonical Q4_K/Q6_K value expressions are unchanged; only the final half
// address changes to [output_octet][k_octet][k_inner][output_inner].
// Target dimensions are divisible by eight, so this permutation adds no bytes.
kernel void fp16_sg_q4k_dequant_pack(
    device const uchar* weight_blocks [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant uint& n_sb [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint row = gid / n_sb;
    if (row >= rows) return;
    const uint b = gid - row * n_sb;
    device const uchar* block = weight_blocks + (row * n_sb + b) * 144;
    // Reconstruction mirrors q4k_linear_tiled exactly: value =
    // dw * sc[idx/32] * code - dm * mn[idx/32].
    const float dw = float(*reinterpret_cast<device const half*>(block));
    const float dm = float(*reinterpret_cast<device const half*>(block + 2));
    uchar sc[8], mn[8];
    q4k_scale_min(block, sc, mn);
    for (uint idx = 0; idx < 256; ++idx) {
        const uint j = idx >> 5;
        const ulong packed_index = ((ulong(row / 8) * (n_sb * 32) + b * 32 + idx / 8) * 64) + (idx % 8) * 8 + row % 8;
        out[packed_index] = half(dw * float(sc[j]) * float(q4k_code(block, idx))
                        - dm * float(mn[j]));
    }
}

// Q6_K twin. Reconstruction mirrors q6k_linear_tiled: the 16 signed sub-scales
// live at byte 192 and each covers 16 values; the super-block scale is the half
// at byte 208; there is no min term.
kernel void fp16_sg_q6k_dequant_pack(
    device const uchar* weight_blocks [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant uint& n_sb [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint row = gid / n_sb;
    if (row >= rows) return;
    const uint b = gid - row * n_sb;
    device const uchar* block = weight_blocks + (row * n_sb + b) * 210;
    const float dw = float(*reinterpret_cast<device const half*>(block + 208));
    device const char* sc = reinterpret_cast<device const char*>(block + 192);
    for (uint idx = 0; idx < 256; ++idx) {
        const ulong packed_index = ((ulong(row / 8) * (n_sb * 32) + b * 32 + idx / 8) * 64) + (idx % 8) * 8 + row % 8;
        out[packed_index] = half(dw * float(int(sc[idx >> 4])) * float(q6k_code(block, idx)));
    }
}


// Direct f32 -> packed half activation conversion. Each physical panel cell
// is initialized, including padding, before any SG projection consumes it.
kernel void fp16_sg_pack_activation_f32(
    device const float* input [[buffer(0)]],
    device half* packed [[buffer(1)]],
    constant uint& kdim [[buffer(2)]],
    constant uint& tokens [[buffer(3)]],
    constant uint& physical_tokens [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= physical_tokens * kdim) return;
    const uint fragment_index = index / 64;
    const uint element = index % 64;
    const uint k_octets = kdim / 8;
    const uint token = (fragment_index / k_octets) * 8 + element / 8;
    const uint k = (fragment_index % k_octets) * 8 + element % 8;
    packed[index] = token < tokens ? half(input[token * kdim + k]) : half(0);
}
