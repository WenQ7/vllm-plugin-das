# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.step5_dsa_norm_rope import (
    short_index_query_rounding_enabled,
)


def test_round_fusion_requires_opt_in_and_short_bf16_geometry(monkeypatch):
    monkeypatch.delenv('VLLM_STEP5_INDEXER_ROUND_IN_NORM', raising=False)
    assert not short_index_query_rounding_enabled(64, 256, torch.bfloat16)
    monkeypatch.setenv('VLLM_STEP5_INDEXER_ROUND_IN_NORM', '1')
    for tokens in (1, 4, 32, 64, 128):
        assert short_index_query_rounding_enabled(tokens, 256, torch.bfloat16)
    for tokens, dim, dtype in (
        (0, 256, torch.bfloat16),
        (129, 256, torch.bfloat16),
        (64, 192, torch.bfloat16),
        (64, 256, torch.float16),
        (64, 256, torch.float32),
    ):
        assert not short_index_query_rounding_enabled(tokens, dim, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/HIP unavailable')
@pytest.mark.parametrize('qpr', [1, 2, 3, 4])
@pytest.mark.parametrize('rotary', [16, 64])
def test_projection_and_scores_match_native_casts_in_full_graph(
    monkeypatch, qpr, rotary,
):
    monkeypatch.setenv('VLLM_STEP4_DSA_FP8_MATH', 'off')
    monkeypatch.setenv('VLLM_STEP5_GROUPED_HEAD_INDEXER', '1')
    monkeypatch.setenv('VLLM_STEP5_INDEXER_ROUND_IN_NORM', '0')
    from vllm_hcu.model_executor.layers import step4_dsa as driver

    requests_count, groups, heads, dim, width = 4, 2, 4, 256, 8192
    tokens = requests_count * qpr
    torch.manual_seed(191)
    packed = torch.randn(
        tokens, groups * heads * dim + 2 * dim + groups * heads,
        device='cuda', dtype=torch.bfloat16,
    )
    qw = torch.randn(dim, device='cuda', dtype=torch.bfloat16)
    kw = torch.randn_like(qw)
    kb = torch.randn_like(qw)
    cos = torch.randn(65536, rotary, device='cuda', dtype=torch.bfloat16)
    sin = torch.randn_like(cos)
    positions = torch.tensor(
        [65532 + j for j in range(qpr)] * requests_count, device='cuda',
    )
    table = torch.randperm(
        requests_count * 1024, device='cuda', dtype=torch.int32,
    ).view(requests_count, 1024)
    requests = torch.arange(
        requests_count, device='cuda', dtype=torch.int32,
    ).repeat_interleave(qpr)
    summary = torch.randn(
        1, requests_count * width, dim, device='cuda', dtype=torch.bfloat16,
    )
    scores = torch.full(
        (groups * tokens, width), -5., device='cuda', dtype=torch.float32,
    )
    indexer = object.__new__(driver.Step4Indexer)
    indexer.owner = SimpleNamespace(
        sparse_indexer_q_norm=SimpleNamespace(weight=qw),
        sparse_indexer_k_norm=SimpleNamespace(weight=kw, bias=kb),
    )
    indexer.geometry = SimpleNamespace(proxy_dim=dim)
    indexer.num_heads = groups * heads
    indexer.num_kv_groups = groups
    indexer.heads_per_group = heads
    indexer.eps = 1e-6
    indexer.slot = 0
    indexer.weight_prescale = 0.5
    indexer.rope_theta = 10000.
    indexer.rope_scaling = None
    indexer.max_position = 65536
    indexer.state = SimpleNamespace(
        rope_tables=lambda **kwargs: (cos, sin),
        summary=summary, regions_per_page=8, _verification_logits=scores,
    )
    metadata = SimpleNamespace(
        num_actual_tokens=tokens, positions=positions, block_table=table,
    )
    schedule = SimpleNamespace(
        num_requests=requests_count, queries_per_request=qpr,
        graph_regions=width, request_indices=requests,
    )

    def project():
        return indexer._project(packed, positions, tokens, packed)

    def reference():
        q, k, z, weights = project()
        out = torch.full_like(scores, -5.)
        driver.short_prefill_indexer_logits(
            q, weights, summary[0], table, requests, positions, out,
            regions_per_page=8, queries_per_request=qpr,
        )
        return driver.round_activations_e4m3(q), k, z, out

    expected_q, expected_k, expected_z, expected_scores = reference()
    monkeypatch.setenv('VLLM_STEP5_INDEXER_ROUND_IN_NORM', '1')

    def fused():
        q, k, z, weights = project()
        out, _, _ = indexer._score_verification(metadata, schedule, q, weights)
        return q, k, z, out

    def forbidden_cast(*args, **kwargs):
        raise AssertionError('scorer launched a redundant E4M3 rounding stage')

    native_cast = driver.round_activations_e4m3
    monkeypatch.setattr(driver, 'round_activations_e4m3', forbidden_cast)
    q, k, z, out = fused()
    for actual, expected in zip(
        (q, k, z, out), (expected_q, expected_k, expected_z, expected_scores),
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, k, z, out = fused()
    for end in (8, 65, 65536):
        packed.normal_()
        positions.copy_(torch.tensor(
            [max(end - qpr + j, 0) for j in range(qpr)] * requests_count,
            device='cuda',
        ))
        scores.fill_(-5.)
        graph.replay()
        actual_scores = out.clone()
        monkeypatch.setenv('VLLM_STEP5_INDEXER_ROUND_IN_NORM', '0')
        monkeypatch.setattr(driver, 'round_activations_e4m3', native_cast)
        expected_q, expected_k, expected_z, expected_scores = reference()
        for actual, expected in zip(
            (q, k, z, actual_scores),
            (expected_q, expected_k, expected_z, expected_scores),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        monkeypatch.setenv('VLLM_STEP5_INDEXER_ROUND_IN_NORM', '1')
        monkeypatch.setattr(driver, 'round_activations_e4m3', forbidden_cast)
