"""Step5 TP4 Pair32 MegaMoE with opt-in uniform DP rows16/32/64/128."""

import os
import torch

from vllm.distributed import get_ep_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

logger = init_logger("vllm.hcu.step5.pair32_megamoe")
ALL_ROWS = os.environ.get("HCU_MEGAMOE_ALL_DECODE") == "1"
ALLOWED_DP_ROWS = (16, 32, 64, 128) if ALL_ROWS else (64,)


def eligible_dp_rows(dp_rows):
    """Use the same collective decision on all EP ranks."""
    return (
        dp_rows is not None
        and len(dp_rows) == 2
        and dp_rows[0] == dp_rows[1]
        and dp_rows[0] in ALLOWED_DP_ROWS
    )


def tp_output_gather():
    tp = get_tp_group()
    gather = getattr(tp, "_step5_pair32_mega_tp_gather_all_rows", None)
    if gather is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MegaMoE TP gather must be initialized before capture")
        from vllm_hcu.distributed.device_communicators.ipc_all_gather import (
            IpcAllGather,
        )

        gather = IpcAllGather(
            tp.cpu_group,
            torch.device("cuda", torch.cuda.current_device()),
            32 * 4096 * 2,
        )
        tp._step5_pair32_mega_tp_gather_all_rows = gather
    return gather


def initialize(runner):
    from deepgemm.mega_step5.mega_moe import get_symm_buffer_for_mega_moe
    from deepgemm.mega_step5.mega_moe import _C

    logger.info_once(
        "Step5 isolated MegaMoE backend=%s variant=%s",
        _C.__file__,
        os.environ.get("DG_MEGA_STEP5_PHASED_VARIANT"),
    )
    layer = runner.routed_experts
    if getattr(layer, "_hcu_step5_fp8_weight_layout", None) != "pair32":
        raise RuntimeError(
            "Step5 MegaMoE requires weights explicitly packed as original Pair32"
        )
    gate = getattr(layer, "_dsv4_channel_deepgemm_masked_w13", None)
    down = getattr(layer, "_dsv4_channel_deepgemm_masked_w2", None)
    if gate is None:
        gate = layer.w13_weight
    if down is None:
        down = layer.w2_weight
    topology = get_ep_group()
    tp = get_tp_group()
    if (topology.world_size, tp.world_size) != (8, 4):
        raise RuntimeError("Step5 Pair32 MegaMoE decode requires EP8 and TP4")
    expected = ((44, 64, 96, 2, 4, 16, 16), (44, 24, 128, 2, 4, 16, 16))
    if (
        (tuple(gate.shape), tuple(down.shape)) != expected
        or gate.dtype != torch.float8_e4m3fn
        or down.dtype != torch.float8_e4m3fn
    ):
        raise RuntimeError(
            f"Step5 MegaMoE expected native Pair32 checkpoint weights, got {gate.shape}, {down.shape}"
        )
    gs = layer.w13_weight_scale.reshape(44, 3072).float().contiguous()
    ds = layer.w2_weight_scale.reshape(44, 4096).float().contiguous()
    buffer = getattr(topology, "_step5_pair32_megamoe_decode_buffer", None)
    if buffer is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Step5 MegaMoE IPC buffers must be initialized before capture"
            )
        expert_map = getattr(layer, "expert_map", None)
        first = topology.rank_in_group * 44
        if expert_map is not None:
            expected_map = torch.full((352,), -1, dtype=expert_map.dtype, device="cpu")
            expected_map[first : first + 44] = torch.arange(
                44, dtype=expert_map.dtype, device="cpu"
            )
            if not torch.equal(expert_map[:352].cpu(), expected_map):
                raise RuntimeError(
                    "Step5 MegaMoE decode requires contiguous physical expert IDs on each rank"
                )
        buffer = get_symm_buffer_for_mega_moe(
            topology.cpu_group, 352, 32, 8, 4096, 1536, peer_memory_mode="ipc",
        )
        topology._step5_pair32_megamoe_decode_buffer = buffer
        logger.info_once("Step5 MegaMoE Pair32 EP8 IPC buffer initialized")
    runner._step5_pair32_mega_state = (buffer, (gate, gs), (down, ds), tp.rank_in_group)
    if ALL_ROWS:
        tp_output_gather()
    return runner._step5_pair32_mega_state


def apply_decode(runner, x, weights, ids):
    state = getattr(runner, "_step5_pair32_mega_state", None)
    capture = torch.cuda.is_current_stream_capturing()
    if state is None and not capture:
        state = initialize(runner)
    if x.ndim != 2 or x.shape[1] != 4096 or x.dtype != torch.bfloat16:
        return None
    kernel = getattr(runner._quant_method, "moe_kernel", None)
    prepare = getattr(kernel, "prepare_finalize", None)
    defer_tp = getattr(getattr(prepare, "buffer", None), "defer_tp_result", False)
    if not defer_tp and not ALL_ROWS:
        return None
    context = get_forward_context()
    dp_rows = context.additional_kwargs.get("step5_pair32_megamoe_dp_rows")
    if dp_rows is None and context.dp_metadata is not None:
        dp_rows = tuple(context.dp_metadata.num_tokens_across_dp_cpu.tolist())
    if capture:
        logger.info_once(
            "Step5 MegaMoE capture gate DP rows=%s mode=%s batch=%s",
            dp_rows,
            context.cudagraph_runtime_mode,
            str(context.batch_descriptor),
        )
    if not eligible_dp_rows(dp_rows):
        return None
    # Eligible global rows are exact graph sizes on every DP. Do not select
    # a local-only fallback if padded rows violate that collective contract.
    if x.shape[0] != dp_rows[0]:
        raise RuntimeError(f"MegaMoE DP rows {dp_rows} do not match {x.shape}")
    if defer_tp and x.shape[0] != 64:
        raise RuntimeError("The existing native TP4 fused output requires64 rows")
    if state is None:
        state = initialize(runner)
    buffer, gate, down, tp_rank = state
    local_rows = x.shape[0] // 4
    start = tp_rank * local_rows
    local = slice(start, start + local_rows)
    from deepgemm.mega_step5.mega_moe import fp8_mega_moe, mega_moe_preprocess

    output = (
        torch.empty_like(x)
        if defer_tp
        else torch.empty((local_rows, 4096), dtype=x.dtype, device=x.device)
    )
    mega_moe_preprocess(x[local], ids[local], weights[local], buffer, local_rows)
    fp8_mega_moe(
        output[:local_rows],
        gate,
        down,
        buffer,
        activation_clamp=7.0,
        tokens_all_ranks=8 * local_rows,
    )
    if capture:
        logger.info_once(
            "Step5 MegaMoE Pair32 decode graph active mode=%s",
            context.cudagraph_runtime_mode,
        )
    else:
        logger.info_once("Step5 MegaMoE Pair32 decode eager active")
    if not defer_tp:
        output = tp_output_gather().gather([output], [local_rows] * 4)[0]
    return output
