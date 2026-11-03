"""Step5-only library dispatch; other owners retain the existing adapter."""
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.ops import gemma_rms_norm as norms


def owner(step5=True):
    return SimpleNamespace(_hcu_step5_owner=step5,
        weight=SimpleNamespace(dtype=torch.float32), variance_epsilon=1e-6,
        forward_native=lambda *args: "native")


def tensor(width=4096, cuda=True, contiguous=True):
    return SimpleNamespace(shape=(64, width), dtype=torch.bfloat16, is_cuda=cuda,
                           is_contiguous=lambda: contiguous, clone=lambda: "original-output")


@pytest.mark.parametrize("width", [192, 4096])
def test_step5_routes_mixed_dtype_without_casting_gamma(monkeypatch, width):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM", "1")
    layer, x, calls = owner(), tensor(width), []
    monkeypatch.setattr(torch.ops.vllm, "hcu_step5_gemma_rms_norm",
                        lambda *args: calls.append(args) or "library")
    assert norms.HcuGemmaRMSNorm.forward_hip(layer, x) == "library"
    assert calls == [(x, layer.weight, layer.variance_epsilon)]
    assert layer.weight.dtype == torch.float32


@pytest.mark.parametrize("kwargs", [dict(width=256), dict(cuda=False), dict(contiguous=False)])
def test_step5_unsupported_layout_keeps_native_semantics(monkeypatch, kwargs):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM", "1")
    monkeypatch.setattr(torch.ops.vllm, "hcu_step5_gemma_rms_norm",
                        lambda *args: pytest.fail("unsupported library invocation"))
    assert norms.HcuGemmaRMSNorm.forward_hip(owner(), tensor(**kwargs)) == "native"


def test_master_off_never_probes_input_or_library(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM", "1")
    assert norms.HcuGemmaRMSNorm.forward_hip(owner(), object()) == "native"


def test_other_models_keep_original_lightop_adapter(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM", "1")
    calls = []
    provider = ModuleType("lightop.norm")
    provider.gemma_rmsnorm = lambda *args, **kwargs: calls.append((args, kwargs))
    provider.gemma_fused_add_rmsnorm = lambda *args: pytest.fail("unexpected fused-add")
    monkeypatch.setitem(sys.modules, "lightop.norm", provider)
    monkeypatch.setattr(torch.ops.vllm, "hcu_step5_gemma_rms_norm",
                        lambda *args: pytest.fail("Step5 dispatch leaked"))
    layer, x = owner(False), tensor()
    assert norms.HcuGemmaRMSNorm.forward_hip(layer, x) == "original-output"
    assert calls == [((x, layer.weight, layer.variance_epsilon), dict(out="original-output"))]
