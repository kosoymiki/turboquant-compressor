"""
TurboQuant fused Triton kernels for decode attention.

Port of donor turboquant/triton_kernels.py with GPL-3.0-or-later license.

Three kernels:
  1. turboquant_mse_score — compute <q_rot, dequant(key)> without materializing keys
  2. turboquant_qjl_score — compute QJL residual correction scores
  3. turboquant_fused_decode — full fused online-softmax attention over TQ-compressed KV

SPDX-License-Identifier: GPL-3.0-or-later
"""

import torch
import triton
import triton.language as tl


# ─── Kernel 1: MSE score ─────────────────────────────────────────────

@triton.jit
def _turboquant_mse_score_kernel(
    Q_ptr, MSE_ptr, NORMS_ptr, CENTROIDS_ptr, OUT_ptr,
    stride_q_bh, stride_q_d,
    stride_m_bh, stride_m_n, stride_m_d,
    stride_n_bh, stride_n_n,
    stride_o_bh, stride_o_n,
    N,
    D: tl.constexpr,
    PACKED_D: tl.constexpr,
    BITS: tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Score = norms[n] * sum_j q_rot[j] * centroid[idx[n,j]]."""
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    scores = tl.zeros([BLOCK_N], dtype=tl.float32)
    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    for byte_idx in range(PACKED_D):
        packed = tl.load(
            MSE_ptr + pid_bh * stride_m_bh + n_offs * stride_m_n + byte_idx * stride_m_d,
            mask=n_mask, other=0
        ).to(tl.int32)

        for sub in range(VALS_PER_BYTE):
            coord_idx = byte_idx * VALS_PER_BYTE + sub
            if coord_idx < D:
                idx = (packed >> (sub * BITS)) & BIT_MASK
                centroid_val = tl.load(CENTROIDS_ptr + idx)
                q_val = tl.load(Q_ptr + pid_bh * stride_q_bh + coord_idx * stride_q_d).to(tl.float32)
                scores += q_val * centroid_val

    norms = tl.load(NORMS_ptr + pid_bh * stride_n_bh + n_offs * stride_n_n,
                    mask=n_mask, other=0.0).to(tl.float32)
    scores = scores * norms

    tl.store(OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n, scores, mask=n_mask)


# ─── Kernel 2: QJL score ─────────────────────────────────────────────

@triton.jit
def _turboquant_qjl_score_kernel(
    Q_SKETCH_ptr, SIGNS_ptr, RES_NORMS_ptr, OUT_ptr,
    stride_qs_bh, stride_qs_d,
    stride_s_bh, stride_s_n, stride_s_d,
    stride_rn_bh, stride_rn_n,
    stride_o_bh, stride_o_n,
    N,
    D: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,
    QJL_SCALE,
    BLOCK_N: tl.constexpr,
):
    """QJL score = qjl_scale * res_norms[n] * sum_j q_sketch[j] * sign[n,j]."""
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    dot = tl.zeros([BLOCK_N], dtype=tl.float32)

    for byte_idx in range(PACKED_D_SIGNS):
        packed = tl.load(
            SIGNS_ptr + pid_bh * stride_s_bh + n_offs * stride_s_n + byte_idx * stride_s_d,
            mask=n_mask, other=0
        ).to(tl.int32)

        for bit in range(8):
            coord_idx = byte_idx * 8 + bit
            if coord_idx < D:
                sign_bit = (packed >> bit) & 1
                sign_val = tl.where(sign_bit == 1, 1.0, -1.0)
                q_val = tl.load(Q_SKETCH_ptr + pid_bh * stride_qs_bh + coord_idx * stride_qs_d).to(tl.float32)
                dot += q_val * sign_val

    res_norms = tl.load(RES_NORMS_ptr + pid_bh * stride_rn_bh + n_offs * stride_rn_n,
                        mask=n_mask, other=0.0).to(tl.float32)
    qjl_scores = dot * res_norms * QJL_SCALE

    existing = tl.load(OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n,
                       mask=n_mask, other=0.0)
    tl.store(OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n,
             existing + qjl_scores, mask=n_mask)


# ─── Kernel 3: Fused decode (online softmax) ─────────────────────────

@triton.jit
def _turboquant_fused_decode_kernel(
    Q_ROT_ptr, Q_SKETCH_ptr,
    MSE_ptr, SIGNS_ptr, NORMS_ptr, RES_NORMS_ptr, CENTROIDS_ptr,
    V_DATA_ptr, V_SCALES_ptr, V_ZEROS_ptr,
    OUT_ptr,
    stride_q_bh, stride_q_d,
    stride_m_bh, stride_m_n, stride_m_d,
    stride_s_bh, stride_s_n, stride_s_d,
    stride_n_bh, stride_n_n,
    stride_rn_bh, stride_rn_n,
    stride_v_bh, stride_v_n, stride_v_d,
    stride_vs_bh, stride_vs_n, stride_vs_g,
    stride_vz_bh, stride_vz_n, stride_vz_g,
    stride_o_bh, stride_o_d,
    N,
    D: tl.constexpr,
    PACKED_D_MSE: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,
    N_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BITS: tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    QJL_SCALE,
    SM_SCALE,
    BLOCK_N: tl.constexpr,
):
    """Full fused decode: scores + online softmax + value aggregation."""
    pid_bh = tl.program_id(0)
    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)

    num_blocks = tl.cdiv(N, BLOCK_N)

    for block_idx in range(num_blocks):
        n_start = block_idx * BLOCK_N
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        # MSE scores
        mse_scores = tl.zeros([BLOCK_N], dtype=tl.float32)
        for byte_idx in range(PACKED_D_MSE):
            packed = tl.load(
                MSE_ptr + pid_bh * stride_m_bh + n_offs * stride_m_n + byte_idx * stride_m_d,
                mask=n_mask, other=0
            ).to(tl.int32)
            for sub in range(VALS_PER_BYTE):
                coord_idx = byte_idx * VALS_PER_BYTE + sub
                if coord_idx < D:
                    idx = (packed >> (sub * BITS)) & BIT_MASK
                    centroid_val = tl.load(CENTROIDS_ptr + idx)
                    q_val = tl.load(Q_ROT_ptr + pid_bh * stride_q_bh + coord_idx * stride_q_d).to(tl.float32)
                    mse_scores += q_val * centroid_val

        key_norms = tl.load(NORMS_ptr + pid_bh * stride_n_bh + n_offs * stride_n_n,
                            mask=n_mask, other=0.0).to(tl.float32)
        mse_scores = mse_scores * key_norms

        # QJL scores
        qjl_dot = tl.zeros([BLOCK_N], dtype=tl.float32)
        for byte_idx in range(PACKED_D_SIGNS):
            packed = tl.load(
                SIGNS_ptr + pid_bh * stride_s_bh + n_offs * stride_s_n + byte_idx * stride_s_d,
                mask=n_mask, other=0
            ).to(tl.int32)
            for bit in range(8):
                coord_idx = byte_idx * 8 + bit
                if coord_idx < D:
                    sign_bit = (packed >> bit) & 1
                    sign_val = tl.where(sign_bit == 1, 1.0, -1.0)
                    q_val = tl.load(Q_SKETCH_ptr + pid_bh * stride_q_bh + coord_idx * stride_q_d).to(tl.float32)
                    qjl_dot += q_val * sign_val

        res_norms = tl.load(RES_NORMS_ptr + pid_bh * stride_rn_bh + n_offs * stride_rn_n,
                            mask=n_mask, other=0.0).to(tl.float32)

        scores = (mse_scores + qjl_dot * res_norms * QJL_SCALE) * SM_SCALE
        scores = tl.where(n_mask, scores, float("-inf"))

        # Online softmax
        m_new = tl.maximum(m_i, tl.max(scores, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha

        # Dequantize values and accumulate
        d_offs = tl.arange(0, D)
        v_quant = tl.load(
            V_DATA_ptr + pid_bh * stride_v_bh + n_offs[:, None] * stride_v_n + d_offs[None, :] * stride_v_d,
            mask=n_mask[:, None], other=0
        ).to(tl.float32)
        g_offs = d_offs // GROUP_SIZE
        v_scale = tl.load(
            V_SCALES_ptr + pid_bh * stride_vs_bh + n_offs[:, None] * stride_vs_n + g_offs[None, :] * stride_vs_g,
            mask=n_mask[:, None], other=1.0
        ).to(tl.float32)
        v_zero = tl.load(
            V_ZEROS_ptr + pid_bh * stride_vz_bh + n_offs[:, None] * stride_vz_n + g_offs[None, :] * stride_vz_g,
            mask=n_mask[:, None], other=0.0
        ).to(tl.float32)
        v_dequant = v_quant * v_scale + v_zero
        acc += tl.sum(p[:, None] * v_dequant, 0)

        m_i = m_new

    acc = acc / l_i
    d_offs = tl.arange(0, D)
    tl.store(OUT_ptr + pid_bh * stride_o_bh + d_offs * stride_o_d, acc)


# ─── Python wrappers ─────────────────────────────────────────────────

def _get_packing_params(bits: int):
    if bits == 1:
        return 1, 8
    elif bits == 2:
        return 2, 4
    elif bits <= 4:
        return 4, 2
    else:
        return 8, 1


def turboquant_mse_score(
    query_rot: torch.Tensor,
    mse_packed: torch.Tensor,
    norms: torch.Tensor,
    centroids: torch.Tensor,
    mse_bits: int,
) -> torch.Tensor:
    """Compute MSE attention scores. Returns (BH, N) logits."""
    if query_rot.dim() == 3:
        query_rot = query_rot.squeeze(1)
    BH, D = query_rot.shape
    N = mse_packed.shape[1]
    packed_d = mse_packed.shape[2]
    eff_bits, vals_per_byte = _get_packing_params(mse_bits)

    out = torch.zeros(BH, N, device=query_rot.device, dtype=torch.float32)
    BLOCK_N = min(128, triton.next_power_of_2(N))
    grid = (BH, triton.cdiv(N, BLOCK_N))

    _turboquant_mse_score_kernel[grid](
        query_rot, mse_packed, norms, centroids, out,
        query_rot.stride(0), query_rot.stride(1),
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        norms.stride(0), norms.stride(1),
        out.stride(0), out.stride(1),
        N=N, D=D, PACKED_D=packed_d,
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte,
        BLOCK_N=BLOCK_N,
    )
    return out


def turboquant_qjl_score(
    q_sketched: torch.Tensor,
    qjl_signs: torch.Tensor,
    residual_norms: torch.Tensor,
    qjl_scale: float,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Compute QJL score contribution. Adds to `out` if provided."""
    if q_sketched.dim() == 3:
        q_sketched = q_sketched.squeeze(1)
    BH, D = q_sketched.shape
    N = qjl_signs.shape[1]
    packed_d_signs = qjl_signs.shape[2]

    if out is None:
        out = torch.zeros(BH, N, device=q_sketched.device, dtype=torch.float32)

    BLOCK_N = min(128, triton.next_power_of_2(N))
    grid = (BH, triton.cdiv(N, BLOCK_N))

    _turboquant_qjl_score_kernel[grid](
        q_sketched, qjl_signs, residual_norms, out,
        q_sketched.stride(0), q_sketched.stride(1),
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        residual_norms.stride(0), residual_norms.stride(1),
        out.stride(0), out.stride(1),
        N=N, D=D, PACKED_D_SIGNS=packed_d_signs,
        QJL_SCALE=qjl_scale,
        BLOCK_N=BLOCK_N,
    )
    return out


def turboquant_fused_decode(
    query: torch.Tensor,
    quantized_key,
    value_quantized,
    Pi: torch.Tensor,
    S: torch.Tensor,
    centroids: torch.Tensor,
    mse_bits: int,
    qjl_scale: float,
    sm_scale: float,
    group_size: int = 32,
) -> torch.Tensor:
    """Fully fused decode: online-softmax attention over TQ-compressed KV."""
    if query.dim() == 3:
        query = query.squeeze(1)
    BH, D = query.shape

    q_rot = torch.matmul(query.float(), Pi.T)
    q_sketch = torch.matmul(query.float(), S.T)

    mse_packed = quantized_key.mse_indices
    qjl_signs = quantized_key.qjl_signs
    norms = quantized_key.norms
    res_norms = quantized_key.residual_norms

    if mse_packed.dim() > 3:
        BH_actual = mse_packed.shape[0] * mse_packed.shape[1]
        mse_packed = mse_packed.reshape(BH_actual, *mse_packed.shape[2:])
        qjl_signs = qjl_signs.reshape(BH_actual, *qjl_signs.shape[2:])
        norms = norms.reshape(BH_actual, -1)
        res_norms = res_norms.reshape(BH_actual, -1)

    N = mse_packed.shape[1]
    packed_d_mse = mse_packed.shape[2]
    packed_d_signs = qjl_signs.shape[2]

    v_data = value_quantized.data
    v_scales = value_quantized.scales
    v_zeros = value_quantized.zeros

    if v_data.dim() > 3:
        v_data = v_data.reshape(BH, N, -1)
        v_scales = v_scales.reshape(BH, N, -1)
        v_zeros = v_zeros.reshape(BH, N, -1)

    N_GROUPS = D // group_size
    eff_bits, vals_per_byte = _get_packing_params(mse_bits)

    out = torch.zeros(BH, D, device=query.device, dtype=torch.float32)
    BLOCK_N = min(64, triton.next_power_of_2(N))

    _turboquant_fused_decode_kernel[(BH,)](
        q_rot, q_sketch,
        mse_packed, qjl_signs, norms, res_norms, centroids,
        v_data, v_scales, v_zeros,
        out,
        q_rot.stride(0), q_rot.stride(1),
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        norms.stride(0), norms.stride(1),
        res_norms.stride(0), res_norms.stride(1),
        v_data.stride(0), v_data.stride(1), v_data.stride(2),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
        v_zeros.stride(0), v_zeros.stride(1), v_zeros.stride(2),
        out.stride(0), out.stride(1),
        N=N, D=D, PACKED_D_MSE=packed_d_mse, PACKED_D_SIGNS=packed_d_signs,
        N_GROUPS=N_GROUPS, GROUP_SIZE=group_size,
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte,
        QJL_SCALE=qjl_scale, SM_SCALE=sm_scale,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out.to(query.dtype)
