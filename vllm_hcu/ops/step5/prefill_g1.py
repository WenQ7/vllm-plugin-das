# SPDX-License-Identifier: Apache-2.0
"""Optional shape-specialized Step5 channel-FP8 prefill projection.

Consumes the existing expert-sorted bridge and keeps the original NK weights,
FP32 scale order, BF16 projection boundary, activation and GEMM2 unchanged.
"""
from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

_announced = False


@functools.lru_cache(maxsize=None)
def _is_gfx938(device_index: int) -> bool:
    props = torch.cuda.get_device_properties(device_index)
    return getattr(props, "gcnArchName", "").split(":", 1)[0] == "gfx938"


@triton.jit
def _prefill_g1_kernel(
    a, w, out, a_scale, w_scale, sorted_ids, expert_ids, nvalid,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr,
):
    pid = tl.program_id(0)
    count = tl.load(nvalid)
    num_m = tl.cdiv(count, BM)
    num_n: tl.constexpr = N // BN
    if pid >= num_m * num_n:
        return
    group = pid // (GM * num_n)
    first_m = group * GM
    group_m = tl.minimum(num_m - first_m, GM)
    pm = first_m + pid % group_m
    pn = (pid % (GM * num_n)) // group_m
    expert = tl.load(expert_ids + pm)
    if expert < 0:
        return
    tl.assume(expert >= 0)
    rows = pm * BM + tl.arange(0, BM)
    token = tl.load(sorted_ids + rows) & 0xFFFFFF
    valid = token < M
    cols = pn * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    # Local routes can fill the entire sorted capacity in a skewed batch;
    # its FP8 input and BF16 output offsets can exceed signed 32-bit bytes.
    ap = a + rows[:, None].to(tl.int64) * K + kk[None, :]
    bp = w + expert * (N * K) + kk[:, None] + cols[None, :] * K
    acc = tl.full((BM, BN), 0.0, tl.float32)
    for _ in range(K // BK):
        av = tl.load(ap)
        bv = tl.load(bp)
        acc = tl.dot(av, bv, acc=acc)
        ap += BK
        bp += BK
    sa = tl.load(a_scale + rows, mask=valid, other=0.0)
    sb = tl.load(w_scale + expert * N + cols)
    projected = (acc * sa[:, None] * sb[None, :]).to(out.dtype.element_ty)
    tl.store(out + rows[:, None].to(tl.int64) * N + cols[None, :],
             projected, mask=valid[:, None])


def try_step5_prefill_g1(
    a: torch.Tensor, w: torch.Tensor, out: torch.Tensor,
    a_scale: torch.Tensor, w_scale: torch.Tensor,
    sorted_ids: torch.Tensor, expert_ids: torch.Tensor, nvalid: torch.Tensor,
    *, num_tokens: int, top_k: int, sort_block_m: int,
) -> bool:
    """Launch the validated full microbatch case, else leave the caller's fallback.

    Eligibility is metadata-only. No device counters are copied to the host and
    no scratch or additional weight copy is allocated here.
    """
    if num_tokens != 65536 or top_k != 8 or sort_block_m != 256:
        return False
    if (a.ndim != 2 or a.shape[1] != 4096 or w.shape != (22, 3072, 4096)
            or out.ndim != 2 or out.shape[1] != 3072
            or a.dtype != torch.float8_e4m3fn or w.dtype != a.dtype
            or out.dtype != torch.bfloat16 or not a.is_cuda):
        return False
    rows = sorted_ids.numel()
    if (rows % 256 or a.shape[0] != rows or out.shape[0] != rows
            or a_scale.numel() != rows or w_scale.numel() != 22 * 3072
            or a_scale.dtype != torch.float32 or w_scale.dtype != torch.float32
            or sorted_ids.ndim != 1 or expert_ids.ndim != 1
            or expert_ids.numel() < rows // 256 or nvalid.numel() != 1
            or any(t.dtype != torch.int32 for t in (sorted_ids, expert_ids, nvalid))):
        return False
    tensors = (a, w, out, a_scale, w_scale, sorted_ids, expert_ids, nvalid)
    if any(t.device != a.device or not t.is_contiguous() for t in tensors):
        return False
    if not _is_gfx938(a.device.index):
        return False
    _prefill_g1_kernel[(triton.cdiv(rows, 256) * 12,)](
        a, w, out, a_scale, w_scale, sorted_ids, expert_ids, nvalid,
        65536, 3072, 4096, 256, 256, 64, 4,
        num_warps=8, num_stages=2, kpack=1, use_block_pingpong=True,
        schedule_hint="none", sched_latency="none", buffer_cache_swizzle=False,
    )
    global _announced
    if not _announced:
        _announced = True
        print("[step5_prefill_g1] active: E22 M65536 sort256 NK channel-FP8", flush=True)
    return True
