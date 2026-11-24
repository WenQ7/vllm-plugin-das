"""CPU contracts for coordinated MegaMoE selection and layout propagation."""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]




@pytest.mark.parametrize("capturing", [False, True])
def test_tp4_finish_uses_the_current_output_rows(capturing):
    path = ROOT / "vllm_hcu/model_executor/layers/fused_moe/moe_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_forward_impl"
    )
    block = next(
        n
        for n in ast.walk(method)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "shared_fusion is not None"
        and any(
            isinstance(k, ast.Name) and k.id == "_step5_tp4_output_kernel"
            for k in ast.walk(n)
        )
    )

    class Tensor:
        def __init__(self, rows):
            self.shape = (rows, 4096)

        def __getitem__(self, index):
            return Tensor(index.stop)

    calls = []
    shared = object()
    space = {
        "torch": SimpleNamespace(
            empty_like=lambda t: Tensor(t.shape[0]),
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: capturing),
        ),
        "self": SimpleNamespace(
            _wait_late_shared_experts=lambda: None, routed_scaling_factor=1
        ),
        "communicator": SimpleNamespace(
            _IS_CAPTURING=False, buffer_ptrs=[123], rank=0, _ptr=1, max_size=65536
        ),
        "buffer": SimpleNamespace(tp_size=4),
        "hidden_states": Tensor(64),
        "shared_output": shared,
        "_step5_tp4_output_kernel": lambda: lambda *args: calls.append(args),
        "_step5_ipc_logger": SimpleNamespace(info_once=lambda *args: None),
    }
    fn = ast.FunctionDef(
        name="run_finish",
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=block.body,
        decorator_list=[],
    )
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), space)
    result = space["run_finish"]()
    assert result[0] is shared and result[1].shape == (64, 4096)
    assert len(calls) == 1 and calls[0][1].shape == (16, 4096)




def function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]






@pytest.mark.parametrize("original", [False, True])
@pytest.mark.parametrize("mode", [None, "1001"])
def test_fallback_g1_and_g2_keep_the_weight_layout(monkeypatch, original, mode):
    fn = function(
        Path(
            "vllm_hcu/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py"
        ),
        "_step5_masked_gemm_options",
        {"os": os, "uses_original_pair32": lambda: original},
    )
    for stage in ("G1", "G2"):
        key = f"HCU_STEP5_DEEPGEMM_MASKED_{stage}_MODE"
        if mode is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, mode)
        options = fn(stage)
        assert options.get("weight_layout") == ("pair32" if original else None)
        assert options.get("config") == ({"MODE": 1001} if mode else None)


@pytest.mark.parametrize("step5", [False, True])
@pytest.mark.parametrize("master", [False, True])
def test_weight_conversion_keeps_other_models_on_upstream_layout(
    monkeypatch, step5, master
):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", str(int(master)))
    monkeypatch.setenv("HCU_STEP5_DEEPGEMM_PAIR32_EPILOGUES", "1")
    monkeypatch.setenv("VLLM_HCU_EP_FUSED_MOE", "0")
    deepgemm = ModuleType("deepgemm")
    deepgemm.marlin_fp8_contiguous_weight = lambda w: torch.full_like(w, 11)
    deepgemm.marlin_fp8_masked_weight = lambda w: torch.full_like(w, 22)
    deepgemm.pack_fp8_weight_pair32 = lambda w: torch.full_like(w, 33)
    monkeypatch.setitem(sys.modules, "deepgemm", deepgemm)
    path = ROOT / "vllm_hcu/patch/worker/op_opt/moe/patch_fp8_oracle.py"
    node = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef)
        and n.name == "hcu_convert_to_fp8_moe_kernel_format"
    )
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "hcu_enum": SimpleNamespace(
            HCU_DEEPGEMM="hcu", AITER="aiter", DEEPGEMM="dg", BATCHED_DEEPGEMM="batched"
        )
    }
    exec(compile(module, str(path), "exec"), namespace)
    layer = SimpleNamespace(weight_block_size=None, _hcu_step5_owner=step5)
    w = torch.zeros((1, 32, 64))
    scale = torch.ones((1, 32))
    result = namespace[node.name]("batched", layer, w, w, scale, scale, None, None)
    assert bool((result[0] == (33 if step5 and master else 22)).all())
    assert getattr(layer, "_hcu_step5_fp8_weight_layout", None) == (
        "pair32" if step5 and master else None
    )
