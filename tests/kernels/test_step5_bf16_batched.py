# SPDX-License-Identifier: Apache-2.0
import pytest
import torch


@pytest.mark.parametrize("n,k", [(3072, 4096), (4096, 1536)])
def test_bf16_live_tiles_match_native_with_dynamic_graph_counts(n, k):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP unavailable")
    from vllm.triton_utils import tl
    from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
        invoke_moe_batched_triton_kernel,
    )
    from vllm_hcu.model_executor.layers.fused_moe.step5_bf16_batched import (
        invoke_step5_bf16_experts,
    )

    torch.manual_seed(107)
    experts, capacity = 4, 160
    a = torch.randn(experts, capacity, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(experts, n, k, device="cuda", dtype=torch.bfloat16)
    # Noncontiguous output views also verify that padding and strides survive.
    reference = torch.full((experts, capacity, n + 8), -7,
                           device="cuda", dtype=torch.bfloat16)
    actual = torch.full_like(reference, -7)
    counts = torch.tensor([0, 1, 33, 129], device="cuda", dtype=torch.int32)
    config = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64)

    def native():
        invoke_moe_batched_triton_kernel(
            a, b, reference[..., :n], counts, tl.bfloat16,
            None, None, None, False, False, False, config, False,
        )

    def compact():
        invoke_step5_bf16_experts(a, b, actual[..., :n], counts, config)

    native()
    compact()
    torch.cuda.synchronize()
    assert torch.equal(actual, reference)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compact()
    # Changing counts across replay exercises zero, partial and multiple tiles.
    for live in ([159, 0, 65, 3], [32, 64, 96, 128], [0, 0, 0, 0]):
        counts.copy_(torch.tensor(live, device="cuda", dtype=torch.int32))
        reference.fill_(-7)
        actual.fill_(-7)
        native()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(actual, reference)
