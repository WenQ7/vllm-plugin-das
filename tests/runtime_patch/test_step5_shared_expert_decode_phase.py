# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS
import pytest
import torch
from vllm.forward_context import BatchDescriptor
from vllm_hcu.model_executor.layers.fused_moe.step5_overlap import is_short_uniform_decode
from vllm_hcu.model_executor.layers.fused_moe.shared_experts import SharedExperts, SharedExpertsOrder

@pytest.mark.parametrize("uniform,tokens,requests,expected", [
    (True,16,16,True),(True,32,16,True),(True,48,16,True),(True,64,16,True),
    (False,64,16,False),(True,512,16,False),(True,65,16,False),
    (True,64,None,False),(True,0,0,False),
])
def test_phase_uses_real_host_descriptor(uniform,tokens,requests,expected):
    context=NS(batch_descriptor=BatchDescriptor(tokens,requests,uniform))
    assert is_short_uniform_decode(context) is expected
    assert is_short_uniform_decode(context) is expected

def test_missing_descriptor_does_not_enable_overlap():
    assert not is_short_uniform_decode(None)
    assert not is_short_uniform_decode(NS(batch_descriptor=None))

@pytest.mark.parametrize("uniform,expected", [
    (False,SharedExpertsOrder.NO_OVERLAP),
    (True,SharedExpertsOrder.MULTI_STREAM_OVERLAPPED),
])
def test_prefill_guard_cannot_be_bypassed_by_mk_internal(monkeypatch,uniform,expected):
    from vllm_hcu.model_executor.layers.fused_moe import shared_experts as module
    import vllm.forward_context
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS","1")
    monkeypatch.setenv("VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH","1")
    monkeypatch.setenv("VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE","1")
    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD","128")
    monkeypatch.setattr(module.current_platform,"is_cuda_alike",lambda:True)
    context=NS(batch_descriptor=BatchDescriptor(64,16,uniform))
    monkeypatch.setattr(vllm.forward_context,"is_forward_context_available",lambda:True)
    monkeypatch.setattr(vllm.forward_context,"get_forward_context",lambda:context)
    item=SharedExperts.__new__(SharedExperts)
    torch.nn.Module.__init__(item)
    item._step5_decode_only=True
    item._stream=object()
    item._mk_can_overlap_shared_experts=lambda:True
    item._moe_config=NS(moe_parallel_config=NS(enable_eplb=False,use_fi_nvl_two_sided_kernels=False))
    assert item._determine_shared_experts_order(torch.empty(64,8))==expected

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP unavailable")
@pytest.mark.parametrize("uniform",[False,True])
def test_native_aux_stream_and_graph_preserve_outputs(monkeypatch,uniform):
    import vllm.config
    import vllm.forward_context
    from vllm_hcu.model_executor.layers.fused_moe import shared_experts as module
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS","1")
    monkeypatch.setenv("VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH","1")
    monkeypatch.setenv("VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE","1")
    monkeypatch.setenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM","0")
    monkeypatch.setenv("VLLM_STEP5_SHARED_EXPERTS_DECODE_ONLY","1")
    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD","128")
    config=NS(model_config=NS(hf_text_config=NS(model_type="step5")))
    monkeypatch.setattr(vllm.config,"get_current_vllm_config_or_none",lambda:config)
    context=NS(batch_descriptor=BatchDescriptor(64,16,uniform))
    monkeypatch.setattr(vllm.forward_context,"is_forward_context_available",lambda:True)
    monkeypatch.setattr(vllm.forward_context,"get_forward_context",lambda:context)
    torch.manual_seed(107)
    layer=torch.nn.Sequential(
        torch.nn.Linear(4096,1536,bias=False),torch.nn.SiLU(),
        torch.nn.Linear(1536,4096,bias=False),
    ).cuda().bfloat16()
    item=SharedExperts(layer,NS(moe_parallel_config=NS(
        enable_eplb=False,use_fi_nvl_two_sided_kernels=False)),False,lambda:False)
    x=torch.randn(64,4096,device="cuda",dtype=torch.bfloat16)
    out=torch.empty_like(x)
    routed=torch.empty_like(x)
    order=(SharedExpertsOrder.MULTI_STREAM_OVERLAPPED if uniform
           else SharedExpertsOrder.NO_OVERLAP)
    assert item._determine_shared_experts_order(x)==order
    assert item.requires_input_preservation(x)==uniform
    assert item.allows_inplace_routed_output(x,x)==(not uniform)
    def call():
        item.maybe_sync_shared_experts_stream(x)
        routed.copy_(x+1)
        item.forward(x,order)
        out.copy_(item.output)
    for _ in range(3):call()
    torch.cuda.synchronize()
    assert torch.equal(out,layer(x))
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):call()
    for _ in range(3):
        x.normal_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out,layer(x))
        assert torch.equal(routed,x+1)
