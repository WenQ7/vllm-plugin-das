# SPDX-License-Identifier: Apache-2.0
"""Packed SlimQuant W4A8 integration with DeepGEMM's Step5 EP8 kernel."""
from __future__ import annotations

import os
import torch

from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group

from .deepgemm_megamoe import DeepGemmMegaMoEMethod, activation_clamp


def select_method(layer, quant_config, moe):
    shape = (moe.num_experts, moe.num_local_experts, moe.hidden_dim,
             moe.intermediate_size_per_partition, moe.experts_per_token)
    if shape != (352, 44, 4096, 1536, 8):
        return None  # Preserve the existing standalone DSV4 INT4 selector.
    config = get_current_vllm_config()
    parallel = config.parallel_config
    if not parallel.enable_expert_parallel or moe.ep_size != 8:
        raise ValueError("DeepGEMM Step5 W4A8 requires EP8")
    if parallel.enable_eplb or parallel.enable_dbo:
        raise ValueError("DeepGEMM Step5 W4A8 does not support EPLB or DBO")
    if (moe.in_dtype != torch.bfloat16 or moe.has_bias or moe.is_lora_enabled
            or moe.pcp_size > 1 or moe.is_sequence_parallel or moe.skip_final_all_reduce):
        raise ValueError("DeepGEMM W4A8 requires BF16, finalized output, and no bias/LoRA/PCP/SP")
    if (getattr(layer, "apply_router_weight_on_input", False)
            or layer.expert_map_manager.num_fused_shared_experts):
        raise ValueError("DeepGEMM W4A8 requires output router weights and separate shared experts")
    activation = str(getattr(moe.activation, "value", moe.activation)).lower()
    if activation not in ("swiglustep", "swiglu_step") or activation_clamp(moe, layer) != 7.0:
        raise ValueError("DeepGEMM Step5 W4A8 requires SWIGLUSTEP with limit 7")
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import should_ignore_layer
    if should_ignore_layer(layer.layer_name, ignore=quant_config.ignore,
                          fused_mapping=quant_config.packed_modules_mapping, use_fnmatch=True):
        raise ValueError("DeepGEMM W4A8 cannot run ignored experts")
    # The checkpoint loader supplies canonical gate rows followed by up rows.
    # Both plain and TILE256 layouts must be chosen before weights are packed.
    if os.environ.get("W4A8_L1_LAYOUT", "tile256") not in ("plain", "tile256"):
        raise ValueError("W4A8_L1_LAYOUT must be plain or tile256")
    return DeepGemmW4A8Method(moe, quant_config)


def _no_reload(*args, **kwargs):
    raise RuntimeError("DeepGEMM W4A8 weights are packed; restart for weight reload or EPLB")


class DeepGemmW4A8Method(DeepGemmMegaMoEMethod):
    _mega_mode = "w4a8"

    def __init__(self, moe, quant_config):
        super().__init__(moe, None)
        self.quant_config = quant_config

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **attrs):
        from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8AiterMoEMethod
        SlimQuantW4A8Int8AiterMoEMethod.create_weights(
            self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **attrs)

    def get_fused_moe_quant_config(self, layer):
        from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
        return FusedMoEQuantConfig.make(torch.int8, w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale, per_act_token_quant=True,
            weight_dtype="int4", gemm1_clamp_limit=7.0)

    @torch.inference_mode()
    def process_weights_after_loading(self, layer):
        from deepgemm import mega
        if self._weights is not None:
            raise RuntimeError("DeepGEMM W4A8 was already packed; restart for reload")
        ep = get_ep_group()
        if ep.world_size != 8:
            raise ValueError("DeepGEMM Step5 W4A8 requires eight physical EP ranks")
        for weight, expected in ((layer.w13_weight, (44, 3072, 2048)),
                                 (layer.w2_weight, (44, 4096, 768))):
            if tuple(weight.shape) != expected or weight.dtype != torch.int8 or not weight.is_contiguous():
                raise ValueError(f"Expected canonical packed INT4 {expected}, got {weight.shape}/{weight.dtype}")
        expert_map = getattr(layer, "expert_map", None)
        if expert_map is not None:
            expected = torch.full((352,), -1, dtype=expert_map.dtype)
            first = ep.rank_in_group * 44
            expected[first:first + 44] = torch.arange(44, dtype=expected.dtype)
            if not torch.equal(expert_map[:352].cpu(), expected):
                raise ValueError("DeepGEMM W4A8 requires contiguous physical expert IDs")
        # SlimQuant stores true signed-INT4 channel scales /16. The public
        # MegaMoE API expects true scales and divides them by 16 for HIPC.
        # Passing checkpoint scales unchanged would shrink each GEMM by 16.
        with torch.inference_mode(False):
            scales = (layer.w13_weight_scale.reshape(44, 3072).float() * 16.0,
                      layer.w2_weight_scale.reshape(44, 4096).float() * 16.0)
            packed = mega.transform_weights_for_mega_moe(
                (layer.w13_weight, scales[0]), (layer.w2_weight, scales[1]),
                mode="w4a8", packed_k=True,
                tile256=os.environ.get("W4A8_L1_LAYOUT", "tile256") == "tile256")
            mega.prepare_w4a8_weight_scales(packed[0][1], packed[1][1])
        layer.w13_weight.data, layer.w2_weight.data = packed[0][0], packed[1][0]
        self._weights = packed
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            parameter = getattr(layer, name)
            if callable(getattr(parameter, "weight_loader", None)):
                parameter.weight_loader = _no_reload
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self._initialize_buffers(layer, mode="w4a8")
