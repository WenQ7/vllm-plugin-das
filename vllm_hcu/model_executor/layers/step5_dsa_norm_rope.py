# SPDX-License-Identifier: Apache-2.0
"""Step5 sparse-indexer Q/K normalization and RoPE in one launch."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_indexer_qk_norm_rope_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    k_bias_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    stride_q_token,
    stride_k_token,
    stride_q_out_token,
    stride_k_out_token,
    stride_cos,
    eps,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """Run Q RMSNorm and K LayerNorm/RoPE on a joint head grid.

    Programs ``[0, NUM_Q_HEADS)`` process Q heads and the last program
    processes the single K head. The arithmetic and bf16 rounding boundaries
    intentionally match ``step4_dsa_kernels.indexer_norm_rope``.
    """
    token = tl.program_id(0)
    lane = tl.program_id(1)
    is_k = lane == NUM_Q_HEADS
    q_head = lane

    dims = tl.arange(0, BLOCK_D)
    dim_ok = dims < head_dim
    q_values = tl.load(
        q_ptr + token * stride_q_token + q_head * head_dim + dims,
        mask=dim_ok & ~is_k,
        other=0.0,
    ).to(tl.float32)
    k_values = tl.load(
        k_ptr + token * stride_k_token + dims,
        mask=dim_ok & is_k,
        other=0.0,
    ).to(tl.float32)
    values = tl.where(is_k, k_values, q_values)

    sum_sq = tl.sum(values * values)
    raw_mean = tl.sum(values) / head_dim
    mean = tl.where(is_k, raw_mean, 0.0)
    variance = tl.where(
        is_k,
        sum_sq / head_dim - raw_mean * raw_mean,
        sum_sq / head_dim,
    )
    rstd = tl.rsqrt(variance + eps)

    q_scale = tl.load(q_weight_ptr + dims, mask=dim_ok, other=0.0).to(tl.float32)
    k_scale = tl.load(k_weight_ptr + dims, mask=dim_ok, other=0.0).to(tl.float32)
    k_bias = tl.load(k_bias_ptr + dims, mask=dim_ok, other=0.0).to(tl.float32)
    scale = tl.where(is_k, k_scale, q_scale + 1.0)
    bias = tl.where(is_k, k_bias, 0.0)
    normalized = (values - mean) * rstd * scale + bias

    unrotated = dim_ok & (dims >= 2 * rotary_dim)
    tl.store(
        q_out_ptr + token * stride_q_out_token + q_head * head_dim + dims,
        normalized.to(q_out_ptr.dtype.element_ty),
        mask=unrotated & ~is_k,
    )
    tl.store(
        k_out_ptr + token * stride_k_out_token + dims,
        normalized.to(k_out_ptr.dtype.element_ty),
        mask=unrotated & is_k,
    )

    half = tl.arange(0, BLOCK_R)
    half_ok = half < rotary_dim
    q_real = tl.load(
        q_ptr + token * stride_q_token + q_head * head_dim + half,
        mask=half_ok & ~is_k,
        other=0.0,
    ).to(tl.float32)
    q_imag = tl.load(
        q_ptr + token * stride_q_token + q_head * head_dim + rotary_dim + half,
        mask=half_ok & ~is_k,
        other=0.0,
    ).to(tl.float32)
    k_real = tl.load(
        k_ptr + token * stride_k_token + half,
        mask=half_ok & is_k,
        other=0.0,
    ).to(tl.float32)
    k_imag = tl.load(
        k_ptr + token * stride_k_token + rotary_dim + half,
        mask=half_ok & is_k,
        other=0.0,
    ).to(tl.float32)
    real = tl.where(is_k, k_real, q_real)
    imag = tl.where(is_k, k_imag, q_imag)

    q_scale_real = tl.load(q_weight_ptr + half, mask=half_ok, other=0.0).to(tl.float32)
    q_scale_imag = tl.load(
        q_weight_ptr + rotary_dim + half, mask=half_ok, other=0.0
    ).to(tl.float32)
    k_scale_real = tl.load(k_weight_ptr + half, mask=half_ok, other=0.0).to(tl.float32)
    k_scale_imag = tl.load(
        k_weight_ptr + rotary_dim + half, mask=half_ok, other=0.0
    ).to(tl.float32)
    k_bias_real = tl.load(k_bias_ptr + half, mask=half_ok, other=0.0).to(tl.float32)
    k_bias_imag = tl.load(
        k_bias_ptr + rotary_dim + half, mask=half_ok, other=0.0
    ).to(tl.float32)
    real = (
        (real - mean)
        * rstd
        * tl.where(is_k, k_scale_real, q_scale_real + 1.0)
        + tl.where(is_k, k_bias_real, 0.0)
    ).to(q_out_ptr.dtype.element_ty).to(tl.float32)
    imag = (
        (imag - mean)
        * rstd
        * tl.where(is_k, k_scale_imag, q_scale_imag + 1.0)
        + tl.where(is_k, k_bias_imag, 0.0)
    ).to(q_out_ptr.dtype.element_ty).to(tl.float32)

    position = tl.load(positions_ptr + token).to(tl.int32)
    cos = tl.load(
        cos_ptr + position * stride_cos + half, mask=half_ok, other=0.0
    ).to(tl.float32)
    sin = tl.load(
        sin_ptr + position * stride_cos + half, mask=half_ok, other=0.0
    ).to(tl.float32)
    rotated_real = real * cos - imag * sin
    rotated_imag = real * sin + imag * cos

    q_base = q_out_ptr + token * stride_q_out_token + q_head * head_dim
    k_base = k_out_ptr + token * stride_k_out_token
    tl.store(q_base + half, rotated_real, mask=half_ok & ~is_k)
    tl.store(q_base + rotary_dim + half, rotated_imag, mask=half_ok & ~is_k)
    tl.store(k_base + half, rotated_real, mask=half_ok & is_k)
    tl.store(k_base + rotary_dim + half, rotated_imag, mask=half_ok & is_k)


def fused_indexer_qk_norm_rope(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
    num_q_heads: int,
    rotary_dim: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if index_k.shape[1] != head_dim:
        raise ValueError(
            "Step5 fused indexer norm expects exactly one K head; "
            f"got shape {tuple(index_k.shape)} and head_dim={head_dim}"
        )
    q_out = torch.empty_like(index_q)
    k_out = torch.empty_like(index_k)
    _fused_indexer_qk_norm_rope_kernel[(index_q.shape[0], num_q_heads + 1)](
        index_q,
        index_k,
        q_out,
        k_out,
        q_norm_weight,
        k_norm_weight,
        k_norm_bias,
        cos,
        sin,
        positions,
        index_q.stride(0),
        index_k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        cos.stride(0),
        eps,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        NUM_Q_HEADS=num_q_heads,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_R=triton.next_power_of_2(rotary_dim),
        # One wave is enough for the 256-wide reduction and avoids the
        # multi-wave synchronization cost that made the old joint launch
        # slower at 64 verification tokens on BW1101.
        num_warps=1,
        num_stages=1,
    )
    return q_out, k_out
