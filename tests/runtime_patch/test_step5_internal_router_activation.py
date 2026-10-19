"""The internal router must pass its explicit activation to the shared MLP."""
from contextlib import nullcontext
from types import SimpleNamespace
import functools
import os
import subprocess
import sys

import pytest
import torch


def _isolated_worker_test(function):
    # Worker replacement modules must bootstrap before upstream MoE imports.
    # Keep this test's registration out of unrelated tests' collection process.
    @functools.wraps(function)
    def run(*args, **kwargs):
        node = kwargs['request'].node.nodeid
        if os.environ.get('VLLM_HCU_ROUTER_ACTIVATION_TEST_CHILD') == node:
            return function(*args, **kwargs)
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', '-xq', node],
            env=dict(os.environ, VLLM_HCU_ROUTER_ACTIVATION_TEST_CHILD=node),
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    return run


class Projection(torch.nn.Module):
    def __init__(self, gate=False):
        super().__init__()
        self.gate = gate
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return (torch.cat([x, x], dim=-1) if self.gate else x), None


class Router(torch.nn.Module):
    def __init__(self, declares_activation, returns_aux):
        super().__init__()
        self.returns_fused_shared_activation = declares_activation
        self.returns_aux = returns_aux

    def forward(self, x):
        return torch.zeros_like(x), x * 7 if self.returns_aux else None


@pytest.mark.parametrize('declares_activation,returns_aux', [(True, True), (True, False), (False, True)])
@_isolated_worker_test
def test_internal_router_handoff_consumes_fresh_activation_only(
    monkeypatch, declares_activation, returns_aux, request
):
    import vllm
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins
    from vllm_hcu.patch.worker import apply_worker_patches

    load_general_plugins()
    apply_worker_patches()
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm_hcu.models.step5_text import Step5MLP

    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: SimpleNamespace(cuda_stream=77))
    shared = Step5MLP.__new__(Step5MLP)
    torch.nn.Module.__init__(shared)
    shared.fused_act = None
    shared.gate_up_proj = Projection(gate=True)
    shared.act_fn = lambda x: x[..., :x.shape[-1] // 2]
    shared.down_proj = Projection()

    runner = MoERunner.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.gate = Router(declares_activation, returns_aux)
    runner._shared_experts = SimpleNamespace(_layer=shared)
    runner._fse_fuse_gate = False
    runner.moe_config = SimpleNamespace(pcp_size=1)
    runner.routed_experts = SimpleNamespace(_ensure_moe_quant_config_init=lambda: None)
    runner._maybe_sync_shared_experts_stream = lambda *args, **kwargs: None
    runner._sequence_parallel_context = lambda: nullcontext()
    runner._maybe_dispatch = lambda x, logits: (x, logits)
    runner._pcp_routing_inputs = lambda *args: nullcontext(None)
    runner._apply_quant_method = lambda **kwargs: (
        shared(kwargs['shared_experts_input']), torch.zeros_like(kwargs['hidden_states'])
    )
    runner._maybe_combine = lambda shared_output, routed_output: shared_output + routed_output

    for value in [1., 2.]:
        x = torch.full((2, 4), value)
        out = runner._forward_impl(x, None, x)
        expected = x * 7 if declares_activation and returns_aux else x
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        assert shared.fused_act is None
    assert shared.gate_up_proj.calls == (0 if declares_activation and returns_aux else 2)
