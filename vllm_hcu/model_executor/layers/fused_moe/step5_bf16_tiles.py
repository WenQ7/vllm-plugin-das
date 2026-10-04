# SPDX-License-Identifier: Apache-2.0
"""Measured BF16 tiles for small Step5 TP2 DeepEP LL batches."""
from __future__ import annotations
import functools
import os
import torch
from vllm.logger import init_logger
logger = init_logger("vllm.hcu.step5_bf16_tiles")
_step5_model_config = None


def configure_step5_bf16_tiles(vllm_config):
    """Keep the model's construction-time config for graph capture and replay."""
    global _step5_model_config
    _step5_model_config = vllm_config
    if os.environ.get("VLLM_STEP5_BF16_TILE_TUNING", "0") == "1":
        from vllm.model_executor.layers.fused_moe.experts import fused_batched_moe
        install_step5_bf16_tiles(fused_batched_moe)
        logger.info("Step5 BF16 tile hook configured: text_model_type=%s",
                    vllm_config.model_config.hf_text_config.model_type)



@functools.lru_cache(maxsize=None)
def _is_gfx938(device):
    return getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").startswith("gfx938")


def install_step5_bf16_tiles(target):
    original = target.invoke_moe_batched_triton_kernel
    if getattr(original, "_step5_bf16_tiles", False):
        return
    reported = False
    rejected = set()

    def report_rejection(reason):
        if reason not in rejected:
            logger.info("Step5 BF16 tile guard: %s", reason)
            rejected.add(reason)

    @functools.wraps(original)
    def invoke(*args, **kwargs):
        nonlocal reported
        if os.environ.get("VLLM_STEP5_BF16_TILE_TUNING", "0") != "1":
            return original(*args, **kwargs)
        A = args[0] if args else kwargs["A"]
        B = args[1] if len(args) > 1 else kwargs["B"]
        if (A.ndim != 3 or A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16
            or (B.shape[1], B.shape[2]) not in ((3072, 4096), (4096, 1536))
            or not _is_gfx938(A.device)):
            return original(*args, **kwargs)
        from vllm.config import get_current_vllm_config_or_none
        from vllm.forward_context import get_forward_context
        cfg = _step5_model_config or get_current_vllm_config_or_none()
        if cfg is None:
            report_rejection("missing model config")
            return original(*args, **kwargs)
        p = cfg.parallel_config
        if (cfg.model_config.hf_text_config.model_type != "step5"
            or p.tensor_parallel_size != 2 or p.data_parallel_size != 4
            or p.pipeline_parallel_size != 1 or p.decode_context_parallel_size != 1
            or p.all2all_backend != "deepep_low_latency"):
            report_rejection(f"config type={cfg.model_config.hf_text_config.model_type}, TP={p.tensor_parallel_size}, DP={p.data_parallel_size}, backend={p.all2all_backend}")
            return original(*args, **kwargs)
        desc = get_forward_context().batch_descriptor
        if desc is None or not 0 < desc.num_tokens <= 64:
            report_rejection(f"batch descriptor={desc}")
            return original(*args, **kwargs)
        config = args[11] if len(args) > 11 else kwargs["config"]
        tuned = dict(config, BLOCK_SIZE_M=32, BLOCK_SIZE_N=64,
                     BLOCK_SIZE_K=64, num_warps=4, num_stages=2)
        if not reported:
            logger.info("Step5 BF16 small-batch tiles enabled: graph_tokens=%d, M32 N64 K64, warps4 stages2",
                        desc.num_tokens)
            reported = True
        if len(args) > 11:
            positional = list(args)
            positional[11] = tuned
            return original(*positional, **kwargs)
        return original(*args, **dict(kwargs, config=tuned))

    invoke._step5_bf16_tiles = True
    target.invoke_moe_batched_triton_kernel = invoke
