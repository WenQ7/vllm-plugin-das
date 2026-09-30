# SPDX-License-Identifier: Apache-2.0
"""Exact GPU selector fallback for the gfx938 Triton register-class error.

The reference Triton kernel fails in the ``tied_rank < remaining`` comparison
with the current DTK compiler. Keep the vendored file intact and substitute
only its selectors when the launcher opts in. IEEE float ordering, tie breaks,
ascending logical IDs, padding and metadata packing match the reference.
"""
import torch


def region_topk_ids(logits, lengths, *, topk, block_r=2048):
    if logits.dtype != torch.float32:
        raise ValueError("selector expects float32 scores")
    rows, width = logits.shape
    out = torch.full((rows, int(topk)), -1, dtype=torch.int32, device=logits.device)
    if width == 0 or topk == 0:
        return out
    ids = torch.arange(width, device=logits.device)
    visible = lengths.to(torch.int64).clamp(0, width)
    bits = logits.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    keys = torch.where(
        (bits & 0x80000000) != 0, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000
    )
    keys = torch.where(ids[None, :] < visible[:, None], keys, -1)
    take = min(int(topk), width)
    ranked = torch.argsort(keys, dim=-1, descending=True, stable=True)[:, :take]
    valid = torch.arange(take, device=logits.device)[None, :] < visible[:, None]
    ranked = torch.where(valid, ranked, width).sort(dim=-1).values
    out[:, :take] = torch.where(ranked < width, ranked, -1).to(torch.int32)
    return out


def region_topk_pack(
    logits,
    lengths,
    q_positions,
    block_table,
    request_indices=None,
    *,
    topk,
    region_size=8,
    regions_per_page,
    block_r=2048,
    _logical_out=None
):
    if block_table.ndim != 2:
        raise ValueError("block_table must be [num_reqs, pages]")
    logical = region_topk_ids(logits, lengths, topk=topk, block_r=block_r)
    safe = logical.clamp_min(0).to(torch.int64)
    if request_indices is None or block_table.shape[0] == 1:
        request_indices = torch.zeros(
            logits.shape[0], dtype=torch.int64, device=logits.device
        )
    pages = block_table[
        request_indices.to(torch.int64)[:, None], safe // regions_per_page
    ]
    physical = pages.to(torch.int64) * regions_per_page + safe % regions_per_page
    valid_tokens = (
        q_positions.to(torch.int64)[:, None] + 1 - safe * region_size
    ).clamp(0, region_size)
    packed = physical | (valid_tokens << 24)
    packed = torch.where((logical >= 0) & (pages >= 0), packed, -1).to(torch.int32)
    if _logical_out is not None:
        _logical_out.copy_(logical)
        return packed, _logical_out
    return packed


def install_torch_selectors():
    from vllm_hcu.model_executor.layers import step4_dsa_kernels as kernels

    kernels.region_topk_ids = region_topk_ids
    kernels.region_topk_pack = region_topk_pack


def sparse_cache_view(cache):
    """Alias the V2 BHNC allocation as K/V, block, token, head, dimension.

    LBNHC keeps a layer's physical token slots contiguous across pages. No
    copy is allowed: writes must remain visible to the runner's KV cache.
    """
    if cache.ndim == 5:
        return cache
    if cache.ndim != 4 or cache.shape[-1] % 2:
        raise ValueError("Expected BHNC cache with packed K/V content")
    blocks, heads, tokens, content = cache.shape
    if cache.stride(0) != tokens * cache.stride(2) or cache.stride(3) != 1:
        raise ValueError("Step5 sparse attention requires VLLM_KV_CACHE_LAYOUT=LBNHC")
    return cache.view(blocks, heads, tokens, 2, content // 2).permute(3, 0, 2, 1, 4)


def flash_cache_view(cache):
    """Bind an empty V2 allocation as the HCU varlen reader's paged K/V.

    Its scatter writer accepts packed content strides, but its reader assumes
    dense token rows. Place K and V in separate sections inside each page so
    both kernels use the same addresses. Whole-page zeroing/copying remains
    valid and no extra memory is allocated.
    """
    if cache.ndim == 5:
        return cache
    if cache.ndim != 4 or cache.shape[-1] % 2:
        raise ValueError("Expected BHNC cache with packed K/V content")
    blocks, heads, tokens, content = cache.shape
    physical = cache.permute(0, 2, 1, 3)
    if not physical.is_contiguous():
        raise ValueError("Step5 FlashAttention requires VLLM_KV_CACHE_LAYOUT=LBNHC")
    return physical.view(blocks, 2, tokens, heads, content // 2)
