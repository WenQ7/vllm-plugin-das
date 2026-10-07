# SPDX-License-Identifier: Apache-2.0
"""Cache declarations, aliasing and compilation contract for DCP2 Attention."""
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm_hcu.model_executor.layers.step5_dcp2_attention import (
    Step5DCP2Attention, Step5DCP2AttentionBackend,
)
from vllm_hcu.v1.step5_dcp2_cache import Step5DCP2SlidingWindowSpec


def make_layer(window):
    # Cache declarations/binding do not require model weights or distributed groups.
    layer = Step5DCP2Attention.__new__(Step5DCP2Attention)
    torch.nn.Module.__init__(layer)
    layer.attn_type = AttentionType.DECODER
    layer.kv_cache_dtype = "auto"
    layer.attn_backend = Step5DCP2AttentionBackend
    layer.sliding_window = window
    layer.num_kv_heads = 4
    layer.head_size = layer.head_size_v = 192
    layer.kv_cache_torch_dtype = torch.bfloat16
    return layer


@pytest.mark.parametrize("window", [None, 512])
def test_all_dcp2_layers_declare_four_cached_heads(window):
    layer = make_layer(window)
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64, skip_page_size_padded=None))
    spec = layer.get_kv_cache_spec(cfg)
    assert isinstance(spec, Step5DCP2SlidingWindowSpec if window else FullAttentionSpec)
    assert spec.block_size == 64
    assert spec.num_kv_heads == 4
    assert spec.head_size == spec.head_size_v == 192
    if window:
        assert not spec.prefix_cacheable
        assert spec.sliding_window == 512


@pytest.mark.parametrize("window", [None, 512])
def test_sparse_and_window_alias_actual_v2_cache_storage(window):
    layer = make_layer(window)
    raw = torch.zeros(2,64,4,384,dtype=torch.bfloat16)
    allocated = raw.permute(0,2,1,3)
    layer.bind_kv_cache(allocated)
    view = layer.kv_cache
    assert view.shape == (2,2,64,4,192)
    assert view.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    key, value = view[0].flatten(0,1), view[1].flatten(0,1)
    assert key.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    assert value.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    key[64,3,191] = 17
    value[63,2,0] = -19
    assert raw[1,0,3,191] == 17
    assert raw[0,63,2,192] == -19


def test_fake_op_retains_local32_head_output_and_dtype():
    with FakeTensorMode():
        query = torch.empty(64,32,192,dtype=torch.bfloat16)
        key = value = torch.empty(64,2,192,dtype=torch.bfloat16)
        positions = torch.empty(64,dtype=torch.int64)
        token = torch.empty(1,dtype=torch.int32)
        result = torch.ops.vllm.step5_dcp2_attention(
            query,key,value,positions,token,"model.layers.3.self_attn.attn")
        assert result.shape == query.shape
        assert result.dtype == query.dtype


def test_piecewise_graph_requires_runtime_op_split(monkeypatch):
    from vllm.config import CUDAGraphMode
    from vllm_hcu.model_executor.layers import step5_dcp2_attention as adapter

    monkeypatch.setattr(adapter, "get_current_vllm_config", lambda: SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.PIECEWISE, splitting_ops=[])))
    with pytest.raises(RuntimeError, match="splitting_ops"):
        Step5DCP2Attention(32,192,192**-.5,num_kv_heads=4)


def test_step5_config_registers_runtime_split_before_compilation(monkeypatch):
    from vllm.config.compilation import CompilationConfig
    from vllm_hcu.step5_config import Step5TextConfig

    monkeypatch.setenv("VLLM_STEP5_DCP2_RUNTIME", "1")
    monkeypatch.delenv("VLLM_STEP4_DISABLE_DSA", raising=False)
    monkeypatch.setattr(CompilationConfig, "_attention_ops", [])
    Step5TextConfig(sparse_config={"enabled": True})
    assert CompilationConfig._attention_ops == [
        "vllm::step4_dsa_index", "vllm::step5_dcp2_attention"]
