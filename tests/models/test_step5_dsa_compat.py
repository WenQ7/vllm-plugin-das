# SPDX-License-Identifier: Apache-2.0
import struct
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_hcu.model_executor.layers.step5_dsa_compat import (
    region_topk_ids,
    region_topk_pack,
)


def test_flash_cache_spec_without_current_config_context():
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backend import AttentionType
    from vllm_hcu.models.step5 import Step5FlashAttentionBackend

    layer = SimpleNamespace(
        attn_type=AttentionType.DECODER,
        kv_cache_dtype="auto",
        attn_backend=Step5FlashAttentionBackend,
        sliding_window=512,
        num_kv_heads=1,
        head_size=192,
        head_size_v=192,
        kv_cache_torch_dtype=torch.bfloat16,
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64, skip_page_size_padded=None)
    )
    spec = Attention.get_kv_cache_spec(layer, config)
    assert (
        spec.block_size,
        spec.sliding_window,
        spec.num_kv_heads,
        spec.head_size,
    ) == (64, 512, 1, 192)


@pytest.mark.parametrize("length", [7, 70])
def test_hcu_flash_cache_prefill_and_decode_match_torch(length):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP unavailable")
    from vllm_hcu.model_executor.layers.step5_dsa_compat import flash_cache_view
    from vllm_hcu.v1.attention.backends.flash_attn import (
        FlashAttentionImpl,
        FlashAttentionMetadata,
        get_flash_attn_version,
    )
    from vllm.v1.attention.backend import AttentionType

    impl = SimpleNamespace(
        attn_type=AttentionType.DECODER,
        kv_cache_dtype="auto",
        num_heads=8,
        head_size=192,
        num_kv_heads=1,
        scale=192 ** -0.5,
        alibi_slopes=None,
        sliding_window=(511, 0),
        logits_soft_cap=0.0,
        kv_sharing_target_layer_name=None,
        vllm_flash_attn_version=get_flash_attn_version(head_size=192),
        batch_invariant_enabled=False,
        supports_quant_query_input=False,
        sinks=None,
        dcp_world_size=1,
        use_pcp=False,
    )
    scale = torch.tensor(1.0, device="cuda")
    layer = SimpleNamespace(_q_scale=scale, _k_scale=scale, _v_scale=scale)
    raw = torch.zeros(2, 64, 1, 384, dtype=torch.bfloat16, device="cuda")
    cache = flash_cache_view(raw.permute(0, 2, 1, 3))
    assert cache.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    torch.manual_seed(19)
    key = torch.randn(length, 1, 192, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    queries = torch.randn(length, 8, 192, dtype=torch.bfloat16, device="cuda")
    FlashAttentionImpl.do_kv_cache_update(
        impl, layer, key, value, cache, torch.arange(length, device="cuda")
    )
    for decode in [False, True]:
        q = queries[-1:] if decode else queries
        count = q.shape[0]
        md = FlashAttentionMetadata(
            num_actual_tokens=count,
            max_query_len=count,
            query_start_loc=torch.tensor([0, count], dtype=torch.int32, device="cuda"),
            max_seq_len=length,
            seq_lens=torch.tensor([length], dtype=torch.int32, device="cuda"),
            block_table=torch.tensor([[0, 1]], dtype=torch.int32, device="cuda"),
            slot_mapping=torch.arange(length - count, length, device="cuda"),
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
        )
        actual = torch.empty_like(q)
        FlashAttentionImpl.forward(
            impl, layer, q, key[-count:], value[-count:], cache, md, actual
        )
        expected = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            key.transpose(0, 1)[None],
            value.transpose(0, 1)[None],
            is_causal=not decode,
            enable_gqa=True,
        )[0].transpose(0, 1)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("heads", [1, 4])
def test_v2_cache_alias_and_sparse_attention_parity(heads):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP unavailable")
    from vllm_hcu.model_executor.layers.step5_dsa_compat import sparse_cache_view
    from vllm_hcu.model_executor.layers.step4_dsa import Step4DSAImpl
    from vllm_hcu.model_executor.layers.step4_dsa_kernels import (
        sparse_attention_prefill,
    )

    # The runner exposes BHNC while physical LBNHC packs K/V per head.
    raw = torch.zeros((2, 8, heads, 384), dtype=torch.bfloat16, device="cuda")
    allocated = raw.permute(0, 2, 1, 3)
    cache = sparse_cache_view(allocated)
    assert cache.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    torch.manual_seed(7)
    key = torch.randn((17, heads, 192), dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    slots = torch.tensor(list(range(16)) + [-1], device="cuda")
    k_flat, v_flat = Step4DSAImpl._write_kv(cache, key, value, slots)
    torch.testing.assert_close(raw[..., :192].flatten(0, 1), key[:16], rtol=0, atol=0)
    torch.testing.assert_close(raw[..., 192:].flatten(0, 1), value[:16], rtol=0, atol=0)
    query = torch.randn((16, heads * 2, 192), dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(16, device="cuda", dtype=torch.int32)
    logits = torch.zeros((heads * 16, 2), device="cuda")
    lengths = ((positions + 8) // 8).repeat(heads)
    logical = torch.empty((heads * 16, 2), dtype=torch.int32, device="cuda")
    packed, logical = region_topk_pack(
        logits,
        lengths,
        positions.repeat(heads),
        torch.tensor([[0, 1]], device="cuda", dtype=torch.int32),
        topk=2,
        regions_per_page=1,
        _logical_out=logical,
    )
    counts = lengths.to(torch.int32)
    out, _ = sparse_attention_prefill(
        query,
        k_flat,
        v_flat,
        packed,
        counts,
        num_kv_groups=heads,
        logical_regions=logical,
        q_positions=positions,
        request_indices=torch.zeros(heads * 16, device="cuda", dtype=torch.int32),
    )
    expected = F.scaled_dot_product_attention(
        query.transpose(0, 1)[None],
        key[:16].transpose(0, 1)[None],
        value[:16].transpose(0, 1)[None],
        is_causal=True,
        enable_gqa=True,
    )[0].transpose(0, 1)
    torch.testing.assert_close(out, expected, rtol=0.02, atol=0.02)


def test_v2_cache_rejects_interleaved_layers():
    from vllm_hcu.model_executor.layers.step5_dsa_compat import sparse_cache_view

    raw = torch.zeros((2, 3, 8, 1, 384))
    with pytest.raises(ValueError, match="LBNHC"):
        sparse_cache_view(raw[:, 1].permute(0, 2, 1, 3))


def _ordered(value):
    bits = struct.unpack("I", struct.pack("f", value))[0]
    return ((~bits) & 0xFFFFFFFF) if bits & 0x80000000 else bits ^ 0x80000000


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("topk", [1, 3, 8])
def test_selection_and_packing_match_scalar_reference(device, topk):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA/HIP unavailable")
    values = [[-1.0, 0.0, -0.0, 2.0, 2.0, -3.0], [0.0] * 6] * 3
    lengths = [-2, 0, 2, 4, 6, 99]
    positions = [0, 11, 20, 35, 44, 49]
    requests = [0, 1, 0, 1, 0, 1]
    table = [[3, 1, -1], [8, -1, 2]]
    expected_ids, expected_pack = [], []
    for row, length, pos, req in zip(values, lengths, positions, requests):
        candidates = range(max(0, min(length, len(row))))
        chosen = sorted(sorted(candidates, key=lambda i: (-_ordered(row[i]), i))[:topk])
        packed = []
        for i in chosen:
            page = table[req][i // 2]
            physical = page * 2 + i % 2
            valid = max(0, min(pos + 1 - i * 8, 8))
            packed.append(physical | (valid << 24) if page >= 0 else -1)
        expected_ids.append(chosen + [-1] * (topk - len(chosen)))
        expected_pack.append(packed + [-1] * (topk - len(packed)))
    logits = torch.tensor(values, dtype=torch.float32, device=device)
    lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    ids = region_topk_ids(logits, lens, topk=topk)
    logical_out = torch.empty_like(ids)
    packed, returned_ids = region_topk_pack(
        logits,
        lens,
        torch.tensor(positions, device=device),
        torch.tensor(table, device=device),
        torch.tensor(requests, device=device),
        topk=topk,
        regions_per_page=2,
        _logical_out=logical_out,
    )
    assert returned_ids is logical_out
    assert ids.cpu().tolist() == expected_ids
    assert returned_ids.cpu().tolist() == expected_ids
    assert packed.cpu().tolist() == expected_pack


def test_shared_block_table_ignores_request_index():
    scores = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    out = region_topk_pack(
        scores,
        torch.tensor([4]),
        torch.tensor([31]),
        torch.tensor([[5, 6]]),
        torch.tensor([999]),
        topk=2,
        regions_per_page=2,
    )
    assert out.tolist() == [[10 | (8 << 24), 11 | (8 << 24)]]
