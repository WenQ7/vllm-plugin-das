# SPDX-License-Identifier: Apache-2.0
"""Opt-in, fixed-shape TP4 output tail for the Step5 MegaMoE adapter.

The persistent kernel still returns one TP token slice. This protocol defers
its gather to the existing native tail, which reduces shared partials, scales
the local routed chunk, and distributes the combined result in one kernel.
"""
from __future__ import annotations
import os
import torch

_ROWS, _WIDTH, _TP = 64, 4096, 4
_SHARED_BYTES = _ROWS * _WIDTH * 2


def megamoe_tp4_tail_communicator(method, tp, rows):
    # Runtime decisions must stay inside step5_moe_forward_output. A symbolic
    # decision outside that op could disagree with _forward_impl's real rows.
    if os.environ.get("HCU_STEP5_TP4_OPAQUE_FINALIZE") != "1":
        return None
    moe = method.moe
    if (getattr(method, "_mega_mode", None) != "w4a8"
            or rows != _ROWS or getattr(method, "_tp_size", 0) != _TP
            or tp.world_size != _TP or method._gather is None
            or getattr(moe, "dp_size", None) != 2
            or getattr(moe, "ep_size", None) != 8
            or getattr(moe, "num_experts", None) != 352
            or getattr(moe, "hidden_dim", None) != _WIDTH
            or getattr(moe, "intermediate_size_per_partition", None) != 1536
            or getattr(moe, "experts_per_token", None) != 8
            or getattr(moe, "in_dtype", None) != torch.bfloat16
            or getattr(moe, "pcp_size", 1) != 1
            or getattr(moe, "is_sequence_parallel", False)):
        return None
    if os.environ.get("VLLM_CUSTOM_ALLREDUCE_ALGO", "2stage") not in ("2stage", "twoshot"):
        return None
    comm = getattr(getattr(tp, "device_communicator", None), "ca_comm", None)
    if (comm is None or comm.disabled or not comm.fully_connected
            or comm.world_size != _TP or comm.max_size < _SHARED_BYTES
            or len(comm.buffer_ptrs) != _TP):
        return None
    return comm


def finish_megamoe_tp4_tail(comm, routed, shared, out, scale, native, *, capturing):
    """Use stable registered shared staging even during Graph capture.

    routed is a full-shape carrier whose first 16 rows contain this TP rank's
    token slice. Its remaining rows are intentionally not read. All TP ranks
    must select the same provider; a violated runtime contract fails closed.
    """
    if shared is None:
        raise ValueError("MegaMoE TP4 fused output requires shared partials")
    for name, tensor in (("routed", routed), ("shared", shared), ("out", out)):
        if (tuple(tensor.shape) != (_ROWS, _WIDTH)
                or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous()
                or tensor.device != routed.device):
            raise ValueError(f"MegaMoE TP4 tail requires contiguous BF16 [64,4096] {name}")
    if not capturing and comm._IS_CAPTURING:
        # Match the existing CustomAR shape-only warmup contract.
        return out
    own = comm.buffer_ptrs[comm.rank]
    # There is no extra D2D copy only when the producer genuinely wrote into
    # that registered allocation. registered=True alone cannot make a normal
    # Graph-pool tensor exportable on this driver.
    staging = 0 if shared.data_ptr() == own else own
    native(comm._ptr, routed[:_ROWS // _TP], shared, out,
           staging, comm.max_size, float(scale))
    return out
