# SPDX-License-Identifier: Apache-2.0
"""GPU checks for the isolated Step5 norm/quant custom-op boundary."""
import pytest
import torch

from vllm_hcu.ops.step5_norm_quant import step5_gemma_norm_fp8_quant

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires native HCU GPU"
)


@pytest.mark.parametrize("layout", range(5))
def test_graph_replay_updates_and_preserves_inputs(layout):
    torch.manual_seed(47)
    x = torch.randn((7, 4096), device="cuda", dtype=torch.float32)
    weight = torch.randn((4096,), device="cuda", dtype=torch.float32) * 0.03
    x_before, weight_before = x.clone(), weight.clone()
    eager = step5_gemma_norm_fp8_quant(x, weight, 1e-5, layout)
    torch.testing.assert_close(x, x_before, rtol=0, atol=0)
    torch.testing.assert_close(weight, weight_before, rtol=0, atol=0)
    assert eager[0].dtype == torch.bfloat16
    assert eager[1].dtype == torch.float8_e4m3fn
    assert eager[2].dtype == torch.float32 and eager[2].shape == (7, 1)
    assert all(y.data_ptr() != x.data_ptr() for y in eager)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step5_gemma_norm_fp8_quant(x, weight, 1e-5, layout)
    # Includes partial row CTAs, changing weights, signs and tiny scales.
    for value in (0., 1., -1., 1e-14, -1e-14):
        x.fill_(value)
        weight.add_(0.0001)
        before = weight.clone()
        graph.replay()
        expected = step5_gemma_norm_fp8_quant(x, weight, 1e-5, layout)
        for actual, reference in zip(captured, expected):
            torch.testing.assert_close(
                actual.float(), reference.float(), rtol=0, atol=0
            )
        torch.testing.assert_close(weight, before, rtol=0, atol=0)
        if value == 0.:
            assert torch.count_nonzero(captured[1].float()) == 0
            assert torch.count_nonzero(captured[2]) == 0
        else:
            assert torch.all(captured[2] > 0)


def test_compiled_dynamic_custom_op_and_empty():
    weight = torch.randn((4096,), device="cuda", dtype=torch.float32) * 0.03
    compiled = torch.compile(
        lambda x, w: step5_gemma_norm_fp8_quant(x, w, 1e-5, 2),
        fullgraph=True, dynamic=True,
    )
    for m in (3, 9, 0):
        x = torch.randn((m, 4096), device="cuda", dtype=torch.float32)
        expected = step5_gemma_norm_fp8_quant(x, weight, 1e-5, 2)
        actual = compiled(x, weight)
        assert actual[2].shape == (m, 1)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a.float(), b.float(), rtol=0, atol=0)


@pytest.mark.parametrize("layout,epsilon", [
    (-1, 1e-5), (5, 1e-5), (0, 0.), (0, float("nan")),
])
def test_invalid_configuration(layout, epsilon):
    x = torch.empty((1, 4096), device="cuda", dtype=torch.float32)
    w = torch.empty((4096,), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError):
        step5_gemma_norm_fp8_quant(x, w, epsilon, layout)


def test_rejects_bf16_norm_weight_and_noncontiguous_input():
    x = torch.randn((2, 4096), device="cuda", dtype=torch.float32)
    w = torch.randn((4096,), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError):
        step5_gemma_norm_fp8_quant(x, w.bfloat16(), 1e-5, 0)
    with pytest.raises(ValueError):
        step5_gemma_norm_fp8_quant(x.t().contiguous().t(), w, 1e-5, 0)
