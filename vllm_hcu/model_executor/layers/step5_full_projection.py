# SPDX-License-Identifier: Apache-2.0
"""Experimental TP4 mixed FP8 QKV / BF16 indexer-gate projection."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import json
import torch
from vllm.utils.torch_utils import direct_register_custom_op
from .step5_mixed_weight_block import pack_mixed_projection_weights


@lru_cache(maxsize=1)
def _tuning():
    return json.loads(Path(__file__).with_name(
        "step5_mixed_projection_tuning.json").read_text())["configs"]


def projection_config(tokens: int, residual_width: int):
    if tokens <= 0 or residual_width not in (16, 1792):
        raise ValueError("Unsupported full Step5 projection geometry")
    table = _tuning()["sparse" if residual_width == 1792 else "dense_gate"]
    bucket = next((n for n in sorted(map(int, table)) if tokens <= n), None)
    if bucket is None:
        raise ValueError("Full Step5 projection is tuned for at most 2048 tokens")
    return table[str(bucket)].copy()


def _mixed_projection(aq: torch.Tensor, activation_scale: torch.Tensor,
                      x: torch.Tensor, qkv_weight: torch.Tensor,
                      weight_scale: torch.Tensor,
                      residual_weight: torch.Tensor) -> torch.Tensor:
    from .step5_mixed_projection_kernel import vllm_step5_mixed_projection
    import triton
    m, k = x.shape
    q = qkv_weight.shape[1]
    r = residual_weight.shape[0]
    if (k != 4096 or q != 3456 or r not in (16, 1792)
            or x.dtype != torch.bfloat16 or aq.dtype != torch.float8_e4m3fn
            or qkv_weight.dtype != torch.float8_e4m3fn
            or residual_weight.dtype != torch.bfloat16
            or activation_scale.dtype != torch.float32
            or weight_scale.dtype != torch.float32
            or not x.is_contiguous() or not aq.is_contiguous()
            or tuple(aq.shape) != (m, k)
            or qkv_weight.stride() != (1, k)
            or not residual_weight.is_contiguous()
            or tuple(residual_weight.shape) != (r, k)
            or not activation_scale.is_contiguous()
            or activation_scale.numel() != m
            or not weight_scale.is_contiguous()
            or tuple(weight_scale.shape) != (q, 1)
            or any(t.device != x.device for t in
                   (aq, activation_scale, qkv_weight, weight_scale, residual_weight))):
        raise ValueError("Unsupported full Step5 projection layout or precision")
    y = torch.empty((m, q + r), device=x.device, dtype=x.dtype)
    if m:
        cfg = projection_config(m, r)
        grid = (triton.cdiv(m, cfg["BMQ"]) * triton.cdiv(q, cfg["BNQ"])
                + triton.cdiv(m, cfg["BMR"]) * triton.cdiv(r, cfg["BNR"]),)
        vllm_step5_mixed_projection[grid](
            aq, activation_scale, x, qkv_weight, weight_scale,
            residual_weight, y, m, q, r, k, PURE_PHASE=0, **cfg)
    return y


def _mixed_projection_fake(aq, activation_scale, x, qkv_weight,
                           weight_scale, residual_weight):
    return torch.empty((x.shape[0],
                        qkv_weight.shape[1] + residual_weight.shape[0]),
                       device=x.device, dtype=x.dtype)


_registered = False


def ensure_registered():
    global _registered
    if not _registered:
        direct_register_custom_op(
            op_name="step5_full_mixed_projection",
            op_func=_mixed_projection, mutates_args=[],
            fake_impl=_mixed_projection_fake)
        _registered = True


@torch.no_grad()
def prepare_full_projection(owner):
    if getattr(owner, "step5_full_projection_ready", False):
        return
    layer = owner.qkv_proj
    k = layer.input_size
    q = owner.q_size + 2 * owner.kv_size
    method = layer.quant_method
    if (not getattr(owner, "qkv_fp8_w8a8", False)
            or k != 4096 or q != 3456 or owner.num_heads != 16
            or not getattr(method, "use_lightop_quant", False)
            or getattr(method, "kernel", None) is None):
        raise ValueError("Full Step5 projection requires finalized TP4 FP8 QKV")
    if not owner.use_head_wise_attn_gate:
        raise ValueError("Full Step5 projection requires the head-wise gate")
    if any(getattr(m, "bias", None) is not None for m in [layer, owner.g_proj]):
        raise ValueError("Full Step5 projection does not support projection biases")
    ranges = {}
    if owner.indexer is not None:
        if not owner.step5_merged_ready or owner.step5_merged_layout != "NK":
            raise RuntimeError("Pack original BF16 residual projections first")
        residual = owner.step5_merged_weight
        if owner.step5_merged_bias is not None:
            raise ValueError("Full Step5 projection does not support indexer biases")
        ranges = dict(owner.step5_merged_offsets)
        if tuple(residual.shape) != (1792, k):
            raise ValueError("Unexpected sparse indexer projection geometry")
    else:
        from .step5_projection import _parallel_weight_nn
        residual = _parallel_weight_nn(owner.g_proj, k, owner.num_heads).t().contiguous()
        ranges["gate"] = (0, owner.num_heads)
    block = pack_mixed_projection_weights(
        layer.weight, residual, layer.weight_scale)
    layer.weight.data = block.qkv_kn
    layer.weight_scale.data = block.qkv_scale
    # Keep all BF16 parameters as views into the same byte allocation.
    # Preserve serialized FP32 index-W, as in the existing residual merger.
    for name, (start, end) in ranges.items():
        m = (getattr(owner, "sparse_indexer_" + name[6:])
             if name.startswith("index_") else owner.g_proj)
        if m.weight.dtype == torch.bfloat16:
            view = block.residual_nk[start:end]
            m.weight.data = (view if name.startswith("index_")
                             or tuple(m.weight.shape) == tuple(view.shape) else view.t())
    owner.step5_merged_weight = block.residual_nk
    owner.step5_full_projection_block = block
    owner.step5_full_projection_offsets = ranges
    owner.step5_full_projection_ready = True
    ensure_registered()


def full_projection(owner, x):
    if not getattr(owner, "step5_full_projection_ready", False):
        raise RuntimeError("Full Step5 projection weights are not prepared")
    b = owner.step5_full_projection_block
    if x.shape[0] == 0:
        y = torch.empty((0, b.plan.qkv_width + b.plan.residual_width),
                        device=x.device, dtype=x.dtype)
    else:
        from vllm_hcu.model_executor.layers.quantization import lightop_fp8_runtime
        x = x.contiguous()
        aq, scale = lightop_fp8_runtime.quantize(
            x, torch.float8_e4m3fn, direct_register_custom_op)
        y = torch.ops.vllm.step5_full_mixed_projection(
            aq, scale, x, b.qkv_kn, b.qkv_scale, b.residual_nk)
    q = b.plan.qkv_width
    ranges = owner.step5_full_projection_offsets
    index = None
    if owner.indexer is not None:
        index = y[:, q + ranges["index_q"][0]:q + ranges["index_w"][1]]
    start, end = ranges["gate"]
    return y[:, :q], index, y[:, q + start:q + end]
