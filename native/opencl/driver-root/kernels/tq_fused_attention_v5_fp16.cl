/**
 * TurboQuant Fused Attention Kernel v5 — tuned for Adreno 730.
 *
 * Over v4 (current stack improvements):
 *   1. async_work_group_copy for query prefetch (OpenCL in Action ch.5)
 *   2. fma() on all fp16 multiply-add paths (ProjectPhysX/OpenCL-Benchmark)
 *   3. Bit-masking instead of % for bank-conflict-free reduction (OpenCL in Action p.250)
 *   4. Prefetched value scales/zeros into __local (KhronosGroup/OpenCL-Guide memory model)
 *   5. Loop-carried accumulator in float for precision (OpenCL_programming.pdf)
 *   6. event_t-based async prefetch overlap with compute
 *   7. cl_khr_subgroups: sub_group_reduce_add eliminates barriers (Adreno 730)
 *      Compile with -DUSE_SUBGROUPS when cl_khr_subgroups is available.
 *      Turnip reports: minSubgroupSize=64, maxSubgroupSize=128, default=128.
 *      WG=64 → partial subgroup (still 1 subgroup, reduce works).
 *      WG=128 → 1 full subgroup = zero overhead, optimal path.
 *
 * Adreno 730 target:
 *   - 2 shader cores, 128-bit ALU (half8 native)
 *   - L1: 64B line, L2: 128B line, 51.2 GB/s LPDDR5
 *   - cl_khr_fp16: YES, fma: native
 *   - cl_khr_subgroups: YES, wave size: 64-128 (subgroupSizeControl)
 *   - Subgroup ops: arithmetic, ballot, shuffle, clustered, rotate
 *   - Local memory: 32KB per compute unit
 *   - maxComputeWorkGroupInvocations: 2048
 *   - shaderFloat16 + shaderInt8 + storageBuffer16BitAccess
 *   - Optimal WG size: 128 (full subgroup) or 64 (vendor CL driver default)
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef USE_SUBGROUPS
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

// Precomputed masks prevent LLVM from creating i3/i4/i5 types via range analysis
__constant uint BIT_MASKS[9] = { 0u, 0x1u, 0x3u, 0x7u, 0xFu, 0x1Fu, 0x3Fu, 0x7Fu, 0xFFu };

static uint extract_bits_v5(__global const uchar* packed, uint coord, uint bits) {
    uint bit_pos = coord * bits;
    uint byte_idx = bit_pos >> 3;
    uint bit_off = bit_pos & 7u;
    uint word = (uint)packed[byte_idx];
    if (bit_off + bits > 8u) word |= ((uint)packed[byte_idx + 1]) << 8;
    // volatile prevents LLVM from narrowing 'bits' to i3 (SPIR-V incompatible)
    volatile uint b = bits;
    return (word >> bit_off) & BIT_MASKS[b];
}

__kernel void tq_fused_attention_v5_fp16(
    __global const half* restrict query_rotated,
    __global const uchar* restrict key_packed,
    __global const half* restrict key_norms,
    __constant const half* restrict centroids,
    __global const uchar* restrict val_packed,
    __global const half* restrict val_scales,
    __global const half* restrict val_zeros,
    __global half* restrict output,
    const int tokens,
    const int dim,
    const int key_bits,
    const int key_packed_stride,
    const int val_bits,
    const int val_packed_stride,
    const int group_size,
    const int n_groups,
    const float sm_scale,
    const float sparse_threshold,
    const int sink_tokens
) {
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);
    const int dim8 = (dim >> 3) << 3;
    const int dim4 = (dim >> 2) << 2;

    if (dim > 256 || key_bits > 8) return;

    // volatile prevents LLVM InstCombine from narrowing to i3 (SPIR-V incompatible)
    volatile int vkey_bits = key_bits;
    volatile int vval_bits = val_bits;
    const int safe_key_bits = vkey_bits;
    const int safe_val_bits = vval_bits;

    // === Centroid LUT → local (small, always fits) ===
    __local half lut[64];
    const int n_centroids = 1 << safe_key_bits;
    for (int i = lid; i < n_centroids; i += local_size)
        lut[i] = centroids[i];

    // === Query prefetch: async_work_group_copy (overlap with LUT init) ===
    __local half q_local[256];
    event_t q_evt = async_work_group_copy(q_local, query_rotated, (size_t)dim, 0);
    wait_group_events(1, &q_evt);

    // === Value scales/zeros prefetch for first token ===
    __local half vs_local[64];  // max 64 groups
    __local half vz_local[64];

    // Online softmax state (float precision — no fp16 accumulation drift)
    float m_i = -INFINITY;
    float l_i = 0.0f;

    // Output accumulator in __local for bank-conflict-free access
    __local half out_local[256];
#ifndef USE_SUBGROUPS
    __local float scratch[256];
#endif
    for (int i = lid; i < dim; i += local_size)
        out_local[i] = (half)0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int n = 0; n < tokens; n++) {
        __global const uchar* k_packed = key_packed + n * key_packed_stride;

        // Prefetch value scales/zeros for this token (async, overlap with key dot)
        const int base_g = n * n_groups;
        event_t vs_evt = async_work_group_copy(vs_local, val_scales + base_g, (size_t)n_groups, 0);
        event_t vz_evt = async_work_group_copy(vz_local, val_zeros + base_g, (size_t)n_groups, 0);

        // === Key dot product: vload8 + fma accumulation ===
        float dot_acc = 0.0f;

        for (int j = lid * 8; j < dim8; j += local_size * 8) {
            half8 q8 = vload8(0, q_local + j);
            half c0 = lut[extract_bits_v5(k_packed, (uint)j,     (uint)safe_key_bits)];
            half c1 = lut[extract_bits_v5(k_packed, (uint)(j+1), (uint)safe_key_bits)];
            half c2 = lut[extract_bits_v5(k_packed, (uint)(j+2), (uint)safe_key_bits)];
            half c3 = lut[extract_bits_v5(k_packed, (uint)(j+3), (uint)safe_key_bits)];
            half c4 = lut[extract_bits_v5(k_packed, (uint)(j+4), (uint)safe_key_bits)];
            half c5 = lut[extract_bits_v5(k_packed, (uint)(j+5), (uint)safe_key_bits)];
            half c6 = lut[extract_bits_v5(k_packed, (uint)(j+6), (uint)safe_key_bits)];
            half c7 = lut[extract_bits_v5(k_packed, (uint)(j+7), (uint)safe_key_bits)];
            // fma chain: more precise than a*b+c*d (donor: ProjectPhysX)
            dot_acc = fma((float)q8.s0, (float)c0, dot_acc);
            dot_acc = fma((float)q8.s1, (float)c1, dot_acc);
            dot_acc = fma((float)q8.s2, (float)c2, dot_acc);
            dot_acc = fma((float)q8.s3, (float)c3, dot_acc);
            dot_acc = fma((float)q8.s4, (float)c4, dot_acc);
            dot_acc = fma((float)q8.s5, (float)c5, dot_acc);
            dot_acc = fma((float)q8.s6, (float)c6, dot_acc);
            dot_acc = fma((float)q8.s7, (float)c7, dot_acc);
        }
        // vload4 remainder
        for (int j = dim8 + lid * 4; j < dim4; j += local_size * 4) {
            half4 q4 = vload4(0, q_local + j);
            half c0 = lut[extract_bits_v5(k_packed, (uint)j,     (uint)safe_key_bits)];
            half c1 = lut[extract_bits_v5(k_packed, (uint)(j+1), (uint)safe_key_bits)];
            half c2 = lut[extract_bits_v5(k_packed, (uint)(j+2), (uint)safe_key_bits)];
            half c3 = lut[extract_bits_v5(k_packed, (uint)(j+3), (uint)safe_key_bits)];
            dot_acc = fma((float)q4.x, (float)c0, dot_acc);
            dot_acc = fma((float)q4.y, (float)c1, dot_acc);
            dot_acc = fma((float)q4.z, (float)c2, dot_acc);
            dot_acc = fma((float)q4.w, (float)c3, dot_acc);
        }
        // Scalar tail
        for (int j = dim4 + lid; j < dim; j += local_size) {
            uint idx = extract_bits_v5(k_packed, (uint)j, (uint)safe_key_bits);
            dot_acc = fma((float)q_local[j], (float)lut[idx], dot_acc);
        }

        // === Workgroup reduction ===
#ifdef USE_SUBGROUPS
        // Adreno 730: WG≤128 = 1 subgroup → single intrinsic, zero barriers.
        // For WG>128 (multi-subgroup), add cross-subgroup reduction here.
        float score = sub_group_reduce_add(dot_acc) * (float)key_norms[n] * sm_scale;
#else
        // Fallback: explicit __local reduction with barriers
        scratch[lid] = dot_acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = local_size >> 1; s > 0; s >>= 1) {
            if (lid < s) scratch[lid] += scratch[lid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float score = scratch[0] * (float)key_norms[n] * sm_scale;
#endif

        // Online softmax update
        float m_new = fmax(score, m_i);
        float alpha = exp(m_i - m_new);
        float p = exp(score - m_new);

        // Sparse skip (protect sink tokens)
        float contrib = p / fma(l_i, alpha, p);
        if (n >= sink_tokens && contrib < sparse_threshold) {
            l_i = fma(l_i, alpha, p);
            m_i = m_new;
            // Rescale output in local (vload8 burst)
            half h_alpha = (half)alpha;
            for (int j = lid * 8; j < dim8; j += local_size * 8) {
                half8 ov = vload8(0, out_local + j);
                vstore8(ov * h_alpha, 0, out_local + j);
            }
            for (int j = dim8 + lid; j < dim; j += local_size)
                out_local[j] *= h_alpha;
            barrier(CLK_LOCAL_MEM_FENCE);
            continue;
        }

        // Wait for scales/zeros prefetch before value dequant
        wait_group_events(1, &vs_evt);
        wait_group_events(1, &vz_evt);

        // === Value dequant + accumulate: fma vectorized, local scales ===
        __global const uchar* v_packed = val_packed + n * val_packed_stride;
        // group_size is power-of-2 → bit shift instead of divide
        // volatile prevents LLVM narrowing to i3 for SPIR-V
        volatile int gs_shift = 31 - clz(group_size);

        for (int j = lid * 4; j < dim4; j += local_size * 4) {
            // Bit-shift division (avoid % on GPU — donor: OpenCL in Action p.250)
            int g0 = j >> gs_shift;
            int g1 = (j+1) >> gs_shift;
            int g2 = (j+2) >> gs_shift;
            int g3 = (j+3) >> gs_shift;

            float s0 = (float)vs_local[g0];
            float s1 = (float)vs_local[g1];
            float s2 = (float)vs_local[g2];
            float s3 = (float)vs_local[g3];
            float z0 = (float)vz_local[g0];
            float z1 = (float)vz_local[g1];
            float z2 = (float)vz_local[g2];
            float z3 = (float)vz_local[g3];

            uint r0 = extract_bits_v5(v_packed, (uint)j,     (uint)safe_val_bits);
            uint r1 = extract_bits_v5(v_packed, (uint)(j+1), (uint)safe_val_bits);
            uint r2 = extract_bits_v5(v_packed, (uint)(j+2), (uint)safe_val_bits);
            uint r3 = extract_bits_v5(v_packed, (uint)(j+3), (uint)safe_val_bits);

            // fma chain: val = raw * scale + zero (donor: ProjectPhysX)
            float v0 = fma((float)r0, s0, z0);
            float v1 = fma((float)r1, s1, z1);
            float v2 = fma((float)r2, s2, z2);
            float v3 = fma((float)r3, s3, z3);

            half4 ov = vload4(0, out_local + j);
            // fma accumulation: out = out * alpha + p * val
            half4 result = (half4)(
                (half)fma((float)ov.x, alpha, p * v0),
                (half)fma((float)ov.y, alpha, p * v1),
                (half)fma((float)ov.z, alpha, p * v2),
                (half)fma((float)ov.w, alpha, p * v3)
            );
            vstore4(result, 0, out_local + j);
        }
        // Scalar tail
        for (int j = dim4 + lid; j < dim; j += local_size) {
            uint raw = extract_bits_v5(v_packed, (uint)j, (uint)safe_val_bits);
            int g = j >> gs_shift;
            float val = fma((float)raw, (float)vs_local[g], (float)vz_local[g]);
            out_local[j] = (half)fma((float)out_local[j], alpha, p * val);
        }

        l_i = fma(l_i, alpha, p);
        m_i = m_new;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Final normalization + writeback: vload8 burst ===
    if (l_i > 0.0f) {
        half inv_l = (half)(1.0f / l_i);
        for (int j = lid * 8; j < dim8; j += local_size * 8) {
            half8 ov = vload8(0, out_local + j);
            vstore8(ov * inv_l, 0, output + j);
        }
        for (int j = dim8 + lid; j < dim; j += local_size)
            output[j] = out_local[j] * inv_l;
    }
}

#ifndef TQ_SPIRV_BUILD
/**
 * Vectorized FWHT kernel v2 (improved) — async prefetch + fma normalize.
 */
__kernel void tq_fwht_v5_fp16(
    __global half* data,
    const int dim
) {
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0);

    if (dim > 256 || (dim & (dim - 1)) != 0) return;

    // Async prefetch from global → local
    __local half buf[256];
    event_t evt = async_work_group_copy(buf, data, (size_t)dim, 0);
    wait_group_events(1, &evt);

    // Butterfly stages (data-dependent, not vectorizable)
    for (int h = 1; h < dim; h <<= 1) {
        for (int i = lid; i < (dim >> 1); i += local_size) {
            // volatile prevents LLVM from narrowing shift to i3 (SPIR-V invalid)
            volatile int shift = 31 - clz((uint)h);
            int block = i >> shift;
            int offset = i & (h - 1);        // i % h via mask
            int j = (block << 1) * h + offset;
            half x = buf[j];
            half y = buf[j + h];
            buf[j] = x + y;
            buf[j + h] = x - y;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Normalize + writeback (vectorized, fma-style)
    half norm = (half)rsqrt((float)dim);
    const int dim8 = (dim >> 3) << 3;
    for (int i = lid * 8; i < dim8; i += local_size * 8) {
        half8 v = vload8(0, buf + i);
        vstore8(v * norm, 0, data + i);
    }
    for (int i = dim8 + lid; i < dim; i += local_size)
        data[i] = buf[i] * norm;
}

/**
 * Bandwidth probe kernel — measures effective memory bandwidth.
 * Donor pattern: ProjectPhysX/OpenCL-Benchmark coalesced read/write.
 */
__kernel void tq_bandwidth_probe(
    __global const float4* restrict src,
    __global float4* restrict dst,
    const int count
) {
    const int gid = get_global_id(0);
    if (gid < count) {
        dst[gid] = src[gid];
    }
}
#endif /* TQ_SPIRV_BUILD */
