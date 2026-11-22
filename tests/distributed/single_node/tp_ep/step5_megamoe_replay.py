"""EP8 correctness: independent Torch FFN, TP reconstruction and repeated HIP Graphs."""
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
import vllm_hcu.hcu_ops
from deepgemm.mega_step5.mega_moe import (
    _C,
    get_symm_buffer_for_mega_moe,
    transform_weights_for_mega_moe,
)
from vllm_hcu.distributed.device_communicators.ipc_all_gather import IpcAllGather
from vllm_hcu.model_executor.layers.fused_moe import step5_pair32_megamoe as adapter

ROOT = Path(
    os.environ.get("STEP5_MEGAMOE_TEST_OUTPUT", "step5-megamoe-results")
).resolve()


def cpu_gather(value):
    value = value.detach().cpu().contiguous()
    pieces = [torch.empty_like(value) for _ in range(8)]
    dist.all_gather(pieces, value)
    return torch.cat(pieces)


def main():
    rank = int(os.environ["LOCAL_RANK"])
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.manual_seed(1812 + rank)  # Probe RNG only; separate processes from inference.
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    assert dist.get_world_size() == 8 and adapter.ALL_ROWS
    for rows in (16, 32, 64, 128):
        assert adapter.eligible_dp_rows((rows, rows))
    for rows in ((16, 32), (0, 0), (8, 8), (132, 132), (64, 60)):
        assert not adapter.eligible_dp_rows(rows)
    groups = [
        dist.new_group(list(range(start, start + 4)), backend="gloo")
        for start in (0, 4)
    ]
    tp_group = groups[rank // 4]
    device = torch.device("cuda", rank)
    ag = IpcAllGather(tp_group, device, 32 * 4096 * 2)
    tp = SimpleNamespace(
        world_size=4,
        rank_in_group=rank % 4,
        cpu_group=tp_group,
        _step5_pair32_mega_tp_gather_all_rows=ag,
    )
    ep = SimpleNamespace(world_size=8, rank_in_group=rank, cpu_group=dist.group.WORLD)
    context = SimpleNamespace(
        additional_kwargs={},
        dp_metadata=None,
        cudagraph_runtime_mode="unit",
        batch_descriptor="unit",
    )
    adapter.get_tp_group = lambda: tp
    adapter.get_ep_group = lambda: ep
    adapter.get_forward_context = lambda: context
    buffer = get_symm_buffer_for_mega_moe(
        dist.group.WORLD, 352, 32, 8, 4096, 1536, peer_memory_mode="ipc"
    )

    def make_weight(n, k):
        raw = torch.empty((44, n, k), dtype=torch.float8_e4m3fn, device=device)
        for expert in raw:
            expert.copy_(
                torch.randn(expert.shape, dtype=torch.bfloat16, device=device) * 0.25
            )
        return raw, torch.full((44, n), 0.015625, device=device)

    raw_gate, gate_sf = make_weight(3072, 4096)
    raw_down, down_sf = make_weight(4096, 1536)
    gate, down = transform_weights_for_mega_moe(
        (raw_gate, gate_sf), (raw_down, down_sf)
    )
    prepare = SimpleNamespace(buffer=SimpleNamespace(defer_tp_result=False))
    runner = SimpleNamespace(
        _step5_pair32_mega_state=(buffer, gate, down, rank % 4),
        _quant_method=SimpleNamespace(
            moe_kernel=SimpleNamespace(prepare_finalize=prepare)
        ),
    )
    records = []
    try:
        for local_rows in (
            16,
            4,
            32,
            8,
        ):  # Grow and shrink the runtime count in one buffer.
            dp_rows = local_rows * 4
            context.additional_kwargs["step5_pair32_megamoe_dp_rows"] = (
                dp_rows,
                dp_rows,
            )
            prepare.buffer.defer_tp_result = dp_rows == 64
            for pattern in ("spread", "concentrated", "clamped"):
                x_local = torch.randn(
                    (local_rows, 4096), device=device, dtype=torch.bfloat16
                )
                if pattern == "clamped":
                    x_local.mul_(32)
                token = torch.arange(local_rows, device=device)[:, None]
                k = torch.arange(8, device=device)[None, :]
                ids_local = ((token * 17 + rank * 37 + k) % 352).long()
                if pattern == "concentrated":
                    ids_local = k.expand(local_rows, 8).contiguous().long()
                weight_local = (
                    torch.arange(1, 9, device=device, dtype=torch.float32)
                    .expand(local_rows, 8)
                    .clone()
                )
                weight_local /= weight_local.sum(-1, keepdim=True)
                all_x = cpu_gather(x_local).to(device)
                all_ids = cpu_gather(ids_local).to(device)
                all_weights = cpu_gather(weight_local).to(device)
                group_slice = slice((rank // 4) * dp_rows, (rank // 4 + 1) * dp_rows)
                x = all_x[group_slice].contiguous()
                ids = all_ids[group_slice].contiguous()
                weights = all_weights[group_slice].contiguous()

                def launch(variant):
                    os.environ["DG_MEGA_STEP5_PHASED_VARIANT"] = variant
                    out = adapter.apply_decode(runner, x, weights, ids)
                    assert out is not None
                    if dp_rows == 64:
                        return ag.gather([out[:local_rows]], [local_rows] * 4)[0]
                    return out

                base = launch("phased").clone()
                candidate = launch("g2-fused-completion")
                torch.cuda.synchronize()
                assert torch.equal(base.view(torch.int16), candidate.view(torch.int16))
                q_input = buffer.x[:local_rows].view(torch.float8_e4m3fn).float()
                input_sf = buffer.x_sf[:local_rows].reshape(local_rows, 1).float()
                dequant = q_input * input_sf
                quant_error = float(
                    (dequant - x_local.float()).norm() / x_local.float().norm()
                )
                assert (
                    quant_error < 0.035 and torch.isfinite(dequant).all()
                ), quant_error
                assert torch.equal(buffer.topk_idx[:local_rows], ids_local)
                q_all = cpu_gather(q_input).to(device)
                sf_all = cpu_gather(input_sf).to(device)
                contribution = torch.zeros(
                    (8 * local_rows, 8, 4096), device=device, dtype=torch.float32
                )
                for expert in range(44):
                    locations = (all_ids == rank * 44 + expert).nonzero()
                    if not locations.numel():
                        continue
                    tokens, routes = locations[:, 0], locations[:, 1]
                    gu = q_all[tokens] @ raw_gate[expert].float().T
                    gu = (
                        (gu * sf_all[tokens] * gate_sf[expert])
                        .to(torch.bfloat16)
                        .float()
                    )
                    g, u = gu.chunk(2, dim=1)
                    activation = (
                        (F.silu(g).clamp_max(7) * u.clamp(-7, 7))
                        .to(torch.bfloat16)
                        .float()
                    )
                    scale = (
                        activation.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 448
                    )
                    aq = (activation / scale).to(torch.float8_e4m3fn).float()
                    y = (
                        (aq @ raw_down[expert].float().T) * scale * down_sf[expert]
                    ).to(torch.bfloat16)
                    contribution[tokens, routes] = y.float()
                whole = contribution.cpu()
                dist.all_reduce(whole)
                reference = torch.zeros((8 * local_rows, 4096), dtype=torch.float32)
                weights_cpu = all_weights.cpu()
                for route in range(8):
                    reference += whole[:, route] * weights_cpu[:, route, None]
                reference = reference.to(torch.bfloat16)[group_slice].to(device)
                error = (candidate.float() - reference.float()).abs()
                rms = float(error.norm() / reference.float().norm().clamp_min(1e-20))
                bound = (
                    0.00355
                    + 0.03 * float(reference.abs().max())
                    + 0.02 * reference.float().abs()
                )
                assert (
                    rms < 0.02
                    and torch.all(error <= bound)
                    and torch.isfinite(candidate).all()
                ), (local_rows, pattern, rms, float(error.max()))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = [launch("g2-fused-completion") for _ in range(4)]
                for iteration in range(3):
                    for out in outputs:
                        out.fill_(float("nan"))
                    buffer.local_buffer[:4].view(torch.int32).fill_(-393216)
                    graph.replay()
                    torch.cuda.synchronize()
                    assert all(
                        torch.equal(out.view(torch.int16), candidate.view(torch.int16))
                        for out in outputs
                    )
                    word = (
                        int(buffer.local_buffer[:4].view(torch.int32).item())
                        & 0xFFFFFFFF
                    )
                    assert (
                        word & 0xFFFF == 0 and word >> 16 == (65530 + 6 * 4) % 65536
                    ), word
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                timings = []
                for _ in range(5):
                    dist.barrier()
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    timings.append(start.elapsed_time(end) * 1000 / 4)
                world_times = [None] * 8
                dist.all_gather_object(world_times, timings)
                record = {
                    "local_tokens": local_rows,
                    "dp_rows": dp_rows,
                    "pattern": pattern,
                    "relative_rms": rms,
                    "input_fp8_relative_rms": quant_error,
                    "max_abs_error": float(error.max()),
                    "eager_graph_bitwise_equal": True,
                    "phased_fused_bitwise_equal": True,
                    "epoch_wrap_passed": True,
                    "tp_output_shape": list(candidate.shape),
                    "timing_scope": "preprocess + MegaMoE + TP gather; synthetic probe, not model TPS",
                    "slowest_rank_median_us": statistics.median(
                        max(t[i] for t in world_times) for i in range(5)
                    ),
                }
                records.append(record)
                if rank == 0:
                    print(json.dumps(record), flush=True)
        results = [None] * 8
        dist.all_gather_object(results, records)
        if rank == 0:
            (ROOT / "unit-results.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "ranks": 8,
                        "cases_per_rank": 12,
                        "extension": str(_C.__file__),
                        "torch": torch.__version__,
                        "reference": "Independent Torch FP8/BF16 G1-SwiGLUStep-quant-G2, routing and TP reconstruction",
                        "results_by_rank": results,
                    },
                    indent=2,
                )
                + "\n"
            )
        dist.barrier()
    finally:
        ag.close()
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
