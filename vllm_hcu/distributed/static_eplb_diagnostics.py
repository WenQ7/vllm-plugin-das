# SPDX-License-Identifier: Apache-2.0
"""Bounded, explicitly armed static-EPLB load diagnostics.

This module never advances EPLB or changes its maps. The native router's existing
record-enabled tensor controls its existing physical-ID counters. Counter rounds
synchronize the device and are excluded from both profiling and formal timing.
No torch import or device action occurs on the inactive fast path.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

ENV_NAME = "HCU_STATIC_EPLB_LOAD_DIAGNOSTICS"
_ENABLED = os.environ.get(ENV_NAME, "0") == "1"
_MAX_STEPS = 64
_MAX_CALLS_PER_STEP = 88 * 4
_ACTIVE = None
_LAST: dict[str, Any] | None = None


def _set_router_observer(enabled: bool) -> None:
    from vllm_hcu.model_executor.layers.fused_moe import router_runtime
    router_runtime._STATIC_LOAD_DIAGNOSTICS_ACTIVE = enabled


def _sync() -> None:
    import torch
    torch.cuda.synchronize()


def _allocate_ring(load_pass, steps: int):
    import torch
    return torch.empty((steps, *load_pass.shape), dtype=load_pass.dtype,
                       device=load_pass.device)


def _host_list(tensor):
    return tensor.detach().to(device="cpu").tolist()


def _ptr(tensor) -> int:
    return int(tensor.data_ptr())


def _rank_identity(worker) -> tuple[int, int]:
    from vllm.distributed.parallel_state import get_ep_group
    group = get_ep_group()
    return int(group.rank_in_group), int(group.world_size)


def _ubatch_id() -> int:
    from vllm.v1.worker.ubatching import dbo_current_ubatch_id
    return int(dbo_current_ubatch_id())


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request(action: str, session: str, max_steps: str):
    if action not in ("begin", "status", "finish"):
        raise ValueError("Static load diagnostic action must be begin, status, or finish")
    if not isinstance(session, str) or (session and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", session)):
        raise ValueError("Diagnostic session must be a fresh lowercase identifier")
    if action == "begin":
        if not session:
            raise ValueError("Diagnostic begin requires a fresh session identifier")
        if not isinstance(max_steps, str) or not re.fullmatch(r"[1-9][0-9]?", max_steps):
            raise ValueError("Diagnostic max_steps must be a string integer in 1..64")
        steps = int(max_steps)
        if steps > _MAX_STEPS:
            raise ValueError("Diagnostic max_steps must be in 1..64")
        return steps
    return None


def _owned_directory(session: str, rank: int, *, probe: bool = False) -> Path:
    raw = os.environ.get("STEP5_STATE_DIR", "")
    case = Path(raw)
    if not raw or not case.is_absolute():
        raise ValueError("STEP5_STATE_DIR must name the existing owned case directory")
    case = case.resolve(strict=True)
    identity = case / "native-service.json"
    if not identity.is_file() or identity.is_symlink():
        raise ValueError("Diagnostic output requires the owned native-service.json case")
    root = case / ("static-eplb-load-status" if probe else "static-eplb-load-diagnostics")
    if root.is_symlink():
        raise ValueError("Diagnostic directory cannot be a symlink")
    root.mkdir(mode=0o700, exist_ok=True)
    if root.resolve() != case / root.name:
        raise ValueError("Diagnostic output escaped its case directory")
    group = root / session
    if group.is_symlink():
        raise ValueError("Diagnostic session cannot be a symlink")
    group.mkdir(mode=0o700, exist_ok=True)
    if group.resolve() != root / session:
        raise ValueError("Diagnostic session escaped its case directory")
    output = group / f"rank-{rank:02d}"
    # Each worker owns one exclusive rank directory. A fresh session is required;
    # sibling workers may create the shared session parent concurrently.
    output.mkdir(mode=0o700, exist_ok=False)
    if output.resolve() != group / output.name:
        raise ValueError("Diagnostic rank directory escaped its session")
    with (output / "owner.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(kind="static_eplb_load_diagnostic", session=session,
                       pid=os.getpid(), rank=rank, case=str(case)), stream)
    return output


def _counter_events(state) -> tuple[int, int]:
    return (int(getattr(state, "_vllm_hcu_offline_transfer_events", 0)),
            int(getattr(state, "_vllm_hcu_offline_rearrangement_events", 0)))


def _runner_state(worker):
    if getattr(worker, "_hcu_static_load_diag_ready", False) is not True:
        raise RuntimeError("Diagnostic begin requires successful normal model warmup")
    if getattr(worker, "_hcu_static_load_diag_dummy_weights", True) is not False:
        raise RuntimeError("Diagnostic begin cannot use dummy weights")
    runner = getattr(worker, "model_runner", None)
    if runner is None or not hasattr(runner, "execute_model_state"):
        raise RuntimeError("Diagnostic requires the current normal model runner")
    if runner.execute_model_state is not None:
        raise RuntimeError("Drain requests before controlling load diagnostics")
    state = getattr(runner, "eplb_state", None)
    if state is None or not getattr(state, "_vllm_hcu_static_source_sha256", ""):
        raise RuntimeError("Diagnostic requires a committed static EPLB map")
    if getattr(state, "is_async", False) is not False or _counter_events(state) != (0, 0):
        raise RuntimeError("Diagnostic requires static no-transfer/no-rearrangement state")
    flag = getattr(state, "should_record_tensor", None)
    if flag is None or flag.numel() != 1:
        raise RuntimeError("Static EPLB has no shared native recording flag")
    return runner, state


def _collect_models(state):
    models, views = [], {}
    for model_state in state.model_states.values():
        key = getattr(model_state, "_hcu_offline_model_key", None)
        if key != "Step5ForCausalLM":
            raise RuntimeError("Diagnostic supports only the static Step5 main model")
        model = model_state.model
        load_pass = model_state.expert_load_pass
        if tuple(load_pass.shape) != (88, 368):
            raise RuntimeError("Diagnostic requires Step5 E23 physical counters[88,368]")
        if str(load_pass.dtype) not in ("torch.int32", "torch.int64"):
            raise RuntimeError("Diagnostic load counters must be integer tensors")
        if getattr(load_pass.device, "type", None) != "cuda":
            raise RuntimeError("Diagnostic requires the normal HCU device counter")
        if not load_pass.is_contiguous() or len(model.moe_layers) != 88:
            raise RuntimeError("Diagnostic Step5 layer/counter layout mismatch")
        mapping = _host_list(model_state.physical_to_logical_map)
        if len(mapping) != 88 or any(len(row) != 368 for row in mapping):
            raise RuntimeError("Diagnostic physical map must be[88,368]")
        per_layer = []
        model_index = len(models)
        for index, layer in enumerate(model.moe_layers):
            layer_state = layer.eplb_state
            view = layer_state.expert_load_view
            if tuple(view.shape) != (368,) or _ptr(view) != _ptr(load_pass[index]):
                raise RuntimeError("Layer native counter does not alias its physical load pass")
            router_owner = getattr(layer, "experts", layer)
            naive = getattr(router_owner, "do_naive_dispatch_combine", None)
            quant = getattr(getattr(router_owner, "routed_experts", None), "quant_method", None)
            internal = getattr(quant, "supports_internal_mk", None)
            per_layer.append(dict(model_layer=index+3, moe_plan_row=index,
                                  do_naive_dispatch_combine=naive,
                                  supports_internal_mk=internal))
            view_key = _ptr(view)
            if view_key in views:
                raise RuntimeError("Diagnostic native layer counter aliases another layer")
            views[view_key] = (model_index, index)
        models.append(dict(state=model_state, key=key, load_pass=load_pass,
                           map_digest=_digest(mapping), physical_to_logical_map=mapping,
                           layer_paths=per_layer, ring=None))
    if len(models) != 1:
        raise RuntimeError("Diagnostic expects exactly one Step5 main model")
    return models, views


def _status(session) -> dict:
    return dict(enabled=_ENABLED, active=bool(session["active"]),
                session=session["name"], rank=session["rank"], world_size=session["world"], pid=os.getpid(),
                recorded_steps=session["steps"], max_steps=session["max_steps"],
                dropped_dummy_or_profile_steps=session["dropped"],
                ring_live=any(x["ring"] is not None for x in session["models"]),
                counter_off_verified=bool(session["off_verified"]),
                static_source_sha256=session["source_sha256"],
                output_dir=str(session["output"]), failure=session.get("failure", ""),
                close_reason=session.get("close_reason", ""), diagnostic_only=True,
                valid_for_performance=False)


def worker_rpc(worker, *, action: str, session: str = "", max_steps: str = "64") -> dict:
    """Thin normal-worker JSON RPC; never returns tensors or model objects."""
    global _ACTIVE, _LAST
    steps = _request(action, session, max_steps)
    if not _ENABLED:
        if action == "status":
            return dict(enabled=False, active=False, ring_live=False, pid=os.getpid(),
                        counter_off_verified=False, diagnostic_only=True)
        raise RuntimeError(f"Set {ENV_NAME}=1 before worker startup to opt in")
    if action == "status":
        if _ACTIVE is not None:
            result = _status(_ACTIVE)
        elif _LAST is not None:
            result = dict(_LAST)
        else:
            _, state = _runner_state(worker)
            rank, world = _rank_identity(worker)
            result = dict(enabled=True, active=False, ring_live=False, pid=os.getpid(),
                          rank=rank, world_size=world, ready=True,
                          static_source_sha256=state._vllm_hcu_static_source_sha256,
                          counter_off_verified=not bool(state.should_record_tensor.item()),
                          diagnostic_only=True, valid_for_performance=False)
        if session:
            # DPLB utility calls may broadcast to all DP engines but return only
            # the first engine's result. These exclusive CPU receipts let the
            # controller verify every actual EP rank without an EP collective.
            output = _owned_directory(session, int(result["rank"]), probe=True)
            path = output / "status.json"
            result.update(probe_session=session, status_path=str(path))
            with path.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
            result["status_sha256"] = _file_digest(path)
        return result
    if action == "finish":
        if _ACTIVE is None:
            raise RuntimeError("No diagnostic session to finish")
        if session != _ACTIVE["name"]:
            raise ValueError("Diagnostic finish session mismatch")
        runner = getattr(worker, "model_runner", None)
        if runner is not _ACTIVE["runner"] or getattr(runner, "eplb_state", None) is not _ACTIVE["state"]:
            raise RuntimeError("Diagnostic worker/EPLB owner changed")
        if runner.execute_model_state is not None:
            raise RuntimeError("Drain requests before controlling load diagnostics")
        return _finish(_ACTIVE)
    if _ACTIVE is not None:
        raise RuntimeError("Finish the previous diagnostic session before begin")
    runner, state = _runner_state(worker)
    rank, world = _rank_identity(worker)
    if world != 16 or not 0 <= rank < world:
        raise RuntimeError("Diagnostic requires the real EP16 group")
    # No counts are armed until readiness, ownership and every model/view have
    # been validated. Synchronizing is intentional in this excluded-only round.
    state.should_record_tensor.fill_(False)
    _sync()
    models, views = _collect_models(state)
    output = _owned_directory(session, rank)
    for model in models:
        model["ring"] = _allocate_ring(model["load_pass"], steps)
        model["load_pass"].zero_()
    _sync()
    value = dict(name=session, state=state, runner=runner, models=models, views=views,
                 rank=rank, world=world, output=output, max_steps=steps,
                 steps=0, dropped=0, active=True, pending=[], metadata=[],
                 started=time.time(), off_verified=False,
                 source_sha256=state._vllm_hcu_static_source_sha256,
                 eplb_step_at_begin=int(getattr(state,"expert_rearrangement_step",0)))
    _ACTIVE = value
    try:
        state.should_record_tensor.fill_(True)
        _set_router_observer(True)
    except BaseException as error:
        _abort(value, error)
        raise
    return _status(value)


def _abort(current, error: BaseException) -> None:
    current["active"] = False
    _set_router_observer(False)
    current["failure"] = f"{type(error).__name__}: {str(error)[:400]}; diagnostic failed closed"
    current["pending"].clear()
    text = str(error).lower()
    fatal = any(x in text for x in ("illegal memory access", "device-side assert", "memory access fault", "hiperrorillegaladdress", "hiperrorlaunchfailure"))
    current["fatal_context"] = fatal
    # Do not issue cleanup kernels against a known broken device context.
    if not fatal:
        try:
            current["state"].should_record_tensor.fill_(False)
        except BaseException:
            pass


def observe_router_path(expert_load_view, topk_ids, num_unpadded_tokens, record_enabled) -> None:
    """Active-only CPU metadata observation before the unchanged native mapper."""
    current = _ACTIVE
    if not _ENABLED or current is None or not current["active"]:
        return
    try:
        if record_enabled is not current["state"].should_record_tensor:
            raise RuntimeError("Diagnostic router uses a different recording flag")
        owner = current["views"].get(_ptr(expert_load_view))
        if owner is None:
            raise RuntimeError("Diagnostic router has an unregistered physical counter")
        if topk_ids.ndim != 2 or topk_ids.shape[1] != 8 or num_unpadded_tokens is None:
            raise RuntimeError("Diagnostic requires routed top8 and native unpadded-token metadata")
        if len(current["pending"]) >= _MAX_CALLS_PER_STEP:
            raise RuntimeError("Diagnostic router-path observation exceeded its bounded step")
        current["pending"].append(dict(model_index=owner[0], moe_plan_row=owner[1],
                                       router_input_rows=int(topk_ids.shape[0]),
                                       topk=8, dbo_ubatch_id=_ubatch_id(),
                                       unpadded_tensor=num_unpadded_tokens))
    except BaseException as error:
        _abort(current, error)
        raise


def on_static_step(state, *, is_dummy: bool = False, is_profile: bool = False) -> None:
    """Bounded post-forward sampling; NEVER calls step/rearrange/transfer."""
    current = _ACTIVE
    if not _ENABLED or current is None or not current["active"] or current["state"] is not state:
        return
    try:
        # DBO may use independent streams. This is deliberately a device-wide
        # fence in a count-only diagnostic round, never a timing measurement.
        _sync()
        if is_dummy or is_profile:
            for model in current["models"]:
                model["load_pass"].zero_()
            current["pending"].clear()
            current["dropped"] += 1
            return
        if not current["pending"]:
            raise RuntimeError("No active router-path observations; do not interpret zero counters")
        calls, unpadded_values = [], {}
        for observed in current["pending"]:
            item = {k:v for k,v in observed.items() if k != "unpadded_tensor"}
            pointer = _ptr(observed["unpadded_tensor"])
            if pointer not in unpadded_values:
                unpadded_values[pointer] = int(observed["unpadded_tensor"].item())
            item["unpadded_tensor_ptr"] = pointer
            item["num_unpadded_tokens"] = unpadded_values[pointer]
            if not 0 <= item["num_unpadded_tokens"] <= item["router_input_rows"]:
                raise RuntimeError("Router unpadded-token metadata is outside its input")
            calls.append(item)
        for model_index, model in enumerate(current["models"]):
            observed_layers = {x["moe_plan_row"] for x in calls if x["model_index"] == model_index}
            if observed_layers != set(range(88)):
                raise RuntimeError("Diagnostic real forward did not cover all88 MoE layers")
            model["ring"][current["steps"]].copy_(model["load_pass"])
            model["load_pass"].zero_()
        current["metadata"].append(dict(local_step=current["steps"], wall_time=time.time(), router_calls=calls))
        current["pending"].clear()
        current["steps"] += 1
        if current["steps"] >= current["max_steps"]:
            state.should_record_tensor.fill_(False)
            current["active"] = False
            _set_router_observer(False)
            current["close_reason"] = "bounded_real_step_limit"
    except BaseException as error:
        _abort(current, error)
        raise


def _finish(current) -> dict:
    global _ACTIVE, _LAST
    if current.get("fatal_context"):
        raise RuntimeError(current["failure"])
    state = current["state"]
    # Disable before any synchronization, CPU export or file operation.
    current["active"] = False
    _set_router_observer(False)
    state.should_record_tensor.fill_(False)
    _sync()
    if bool(state.should_record_tensor.item()):
        raise RuntimeError("Native load recording flag did not turn off")
    current["off_verified"] = True
    try:
        if current.get("failure"):
            raise RuntimeError(current["failure"])
        return _export_finish(current)
    except BaseException as error:
        result = _status(current)
        result.update(active=False, ring_live=False, counter_off_verified=True,
                      failure=f"Export/audit failed: {type(error).__name__}: {str(error)[:300]}")
        _LAST = result
        raise
    finally:
        # Only reached after the device flag was confirmed off. Never free
        # diagnostic GPU buffers as cleanup against a known broken context.
        for model in current["models"]:
            model["ring"] = None
        current["pending"].clear()
        current["views"].clear()
        current["metadata"].clear()
        _ACTIVE = None


def _export_finish(current) -> dict:
    global _LAST
    state = current["state"]
    if _counter_events(state) != (0,0) or int(getattr(state,"expert_rearrangement_step",0)) != current["eplb_step_at_begin"]:
        raise RuntimeError("Static diagnostic changed EPLB lifecycle counters")
    if any(_digest(_host_list(m["state"].physical_to_logical_map)) != m["map_digest"] for m in current["models"]):
        raise RuntimeError("Static physical map changed during diagnostic")
    import numpy as np
    arrays, model_metadata, validation = {}, [], True
    for index, model in enumerate(current["models"]):
        array = model["ring"][:current["steps"]].detach().to(device="cpu").numpy()
        key = f"physical_counts_{index}"
        arrays[key] = array
        expected = np.zeros((current["steps"],88), dtype=np.int64)
        for step, item in enumerate(current["metadata"]):
            for call in item["router_calls"]:
                if call["model_index"] == index:
                    expected[step,call["moe_plan_row"]] += call["num_unpadded_tokens"] * call["topk"]
        valid = bool(np.all(array >= 0) and np.array_equal(array.sum(2),expected))
        validation = validation and valid
        path_ambiguous = any(x["do_naive_dispatch_combine"] is not False for x in model["layer_paths"])
        model_metadata.append(dict(key=model["key"], array_key=key, shape=list(array.shape),
                                   map_digest=model["map_digest"], physical_to_logical_map=model["physical_to_logical_map"],
                                   layer_paths=model["layer_paths"], routed_token_totals_match=valid,
                                   expected_routed_totals=expected.tolist(),
                                   aggregation_contract=("source_local_single_count" if not path_ambiguous else
                                                         "ambiguous_naive_or_unknown_path_do_not_sum_or_divide_by_dp")))
    counts_path, metadata_path = current["output"] / "physical-loads.npz", current["output"] / "metadata.json"
    if current["output"].is_symlink() or current["output"].resolve() != current["output"]:
        raise RuntimeError("Diagnostic owned output directory changed")
    with counts_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    metadata = dict(schema_version=1, diagnostic_only=True, valid_for_performance=False,
                    session=current["name"], rank=current["rank"], world_size=current["world"], pid=os.getpid(),
                    started=current["started"], finished=time.time(), recorded_steps=current["steps"],
                    dropped_dummy_or_profile_steps=current["dropped"], max_steps=current["max_steps"],
                    counter_off_verified=True, static_source_sha256=current["source_sha256"],
                    transfer_events=0, rearrangement_events=0, eplb_step_unchanged=True,
                    device_synchronized_each_recorded_step=True, models=model_metadata,
                    step_metadata=current["metadata"], routed_token_totals_match=validation,
                    counts_path=str(counts_path), counts_sha256=_file_digest(counts_path),
                    no_automatic_dp_normalization=True)
    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    result = _status(current)
    result.update(active=False, ring_live=False, counter_off_verified=True,
                  counts_path=str(counts_path), counts_sha256=_file_digest(counts_path),
                  metadata_path=str(metadata_path), metadata_sha256=_file_digest(metadata_path),
                  routed_token_totals_match=validation)
    _LAST = result
    return dict(result)


__all__ = ["worker_rpc", "observe_router_path", "on_static_step"]
