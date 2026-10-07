
# SPDX-License-Identifier: Apache-2.0
"""Opt-in native paged Decode for Step5's causal sparse regions.

Each query/KV-group selection becomes BF16 pages in the shared arena.
The installed gfx938 operator owns Attention and split reduction.
It returns no LSE, so this entry point is only for DCP1.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl
from vllm.logger import init_logger
logger = init_logger("vllm.hcu.step5.native_sparse_decode")
_MAX_TOKENS = 64
_TOPK = 513
_PAGES = 65

def eligible(query, key, value, packed, counts, output) -> bool:
    tokens = query.shape[0]
    return (
        0 < tokens <= _MAX_TOKENS and query.shape == (tokens, 32, 192)
        and query.dtype == torch.bfloat16
        and key.ndim == value.ndim == 3
        and key.shape == value.shape and key.shape[1:] == (2, 192)
        and key.dtype == value.dtype == torch.float8_e4m3fn
        and key.stride(2) == value.stride(2) == 1
        and packed.shape == (2 * tokens, _TOPK)
        and packed.dtype == counts.dtype == torch.int32
        and counts.shape == (2 * tokens,)
        and packed.is_contiguous() and counts.is_contiguous()
        and query.is_contiguous() and output.is_contiguous()
        and output.shape == query.shape and output.dtype == query.dtype
        and all(t.device == query.device for t in (key, value, packed, counts, output))
    )

def _workspace(tokens: int):
    from vllm.v1.worker.workspace import current_workspace_manager
    return current_workspace_manager().get_simultaneous(
        ((2 * tokens * _PAGES, 64, 1, 192), torch.bfloat16),
        ((2 * tokens * _PAGES, 64, 1, 192), torch.bfloat16),
        ((2 * tokens,), torch.int32),
        ((2 * tokens, _PAGES), torch.int32),
        ((2 * tokens + 1,), torch.int32),
    )

def reserve_workspace() -> None:
    """Reserve the largest supported shape during metadata-free profiling."""
    _workspace(_MAX_TOKENS)

@triton.jit
def _pack_sparse_pages(key, value, packed, counts, dense_key, dense_value,
                       lengths, table, cu_query,
                       K_TOKEN, K_GROUP, V_TOKEN, V_GROUP,
                       TOKENS: tl.constexpr, K_SCALE: tl.constexpr,
                       V_SCALE: tl.constexpr, BLOCK_T: tl.constexpr):
    tile = tl.program_id(0)
    token = tl.program_id(1)
    group = tl.program_id(2)
    row = group * TOKENS + token
    destination = token * 2 + group
    count = tl.minimum(tl.maximum(tl.load(counts + row), 0), 513)
    offsets = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    dims = tl.arange(0, 256)
    regions = offsets // 8
    word = tl.load(packed + row * 513 + regions, mask=regions < count, other=-1)
    physical = word & 0x00FFFFFF
    valid = (word >> 24) & 255
    live = (regions < count) & (word >= 0) & (offsets % 8 < valid)
    source = tl.where(live, physical * 8 + offsets % 8, 0).to(tl.int64)
    mask = live[:, None] & (dims[None, :] < 192)
    k = tl.load(key + source[:, None] * K_TOKEN + group * K_GROUP + dims[None, :],
                mask=mask, other=0.0).to(tl.float32)
    v = tl.load(value + source[:, None] * V_TOKEN + group * V_GROUP + dims[None, :],
                mask=mask, other=0.0).to(tl.float32)
    target = (destination * 4160 + offsets[:, None]) * 192 + dims[None, :]
    write = (offsets[:, None] < 4160) & (dims[None, :] < 192)
    tl.store(dense_key + target, (k * K_SCALE).to(tl.bfloat16), mask=write)
    tl.store(dense_value + target, (v * V_SCALE).to(tl.bfloat16), mask=write)
    if tile == 0:
        tail = tl.load(packed + row * 513 + count - 1, mask=count > 0, other=0)
        length = tl.maximum(count - 1, 0) * 8 + ((tail >> 24) & 255)
        # Empty padded rows attend to one zero KV, producing finite zero output.
        tl.store(lengths + destination, tl.maximum(length, 1))
        pages = tl.arange(0, 128)
        tl.store(table + destination * 65 + pages, destination * 65 + pages,
                 mask=pages < 65)
        tl.store(cu_query + destination, destination)
        if destination == TOKENS * 2 - 1:
            tl.store(cu_query + destination + 1, destination + 1)

def sparse_attention_native_paged(query, key, value, packed, counts, *,
                                  output, softmax_scale=None,
                                  k_scale=1.0, v_scale=1.0):
    if not eligible(query, key, value, packed, counts, output):
        raise ValueError("Unsupported Step5 native sparse Decode geometry")
    from flash_attn import flash_attn_varlen_func
    tokens = query.shape[0]
    k, v, lengths, table, cu_query = _workspace(tokens)
    logger.info_once(
        "Step5 sparse Decode uses native HCU paged Attention: "
        "BF16 sparse pages in shared workspace; DCP1; max_tokens=64",
        scope="process",
    )
    _pack_sparse_pages[(130, tokens, 2)](
        key, value, packed, counts, k, v, lengths, table, cu_query,
        key.stride(0), key.stride(1), value.stride(0), value.stride(1),
        TOKENS=tokens, K_SCALE=k_scale, V_SCALE=v_scale,
        BLOCK_T=32, num_warps=4,
    )
    result = flash_attn_varlen_func(
        query.reshape(tokens * 2, 16, 192), k, v,
        cu_seqlens_q=cu_query, max_seqlen_q=1, seqused_k=lengths,
        max_seqlen_k=4104, block_table=table,
        out=output.reshape(tokens * 2, 16, 192),
        softmax_scale=softmax_scale if softmax_scale is not None else 192 ** -0.5,
        causal=False, layout="bshd", return_softmax_lse=False,
    )
    if result.data_ptr() != output.data_ptr():
        raise RuntimeError("Native sparse Decode did not use its provided output")
    return output, None
