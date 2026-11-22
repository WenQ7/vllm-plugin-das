"""CPU contracts for coordinated MegaMoE selection and layout propagation."""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_forward_context_helper_has_its_runtime_imports(monkeypatch):
    path = ROOT / "vllm_hcu/forward_context_runtime.py"
    spec = importlib.util.spec_from_file_location("_step5_context_import_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("HCU_STEP5_PAIR32_MEGAMOE_DECODE", "1")
    monkeypatch.setenv("HCU_STEP5_MEGAMOE_ALL_DECODE", "1")
    assert module._step5_megamoe_dp_rows(16, SimpleNamespace(tolist=lambda: [16,16])) == (16,16)


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


@pytest.mark.parametrize("all_rows", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rows", [0, 16, 32, 64, 128, 129])
def test_context_preserves_coordinated_counts(monkeypatch, all_rows, enabled, rows):
    monkeypatch.setenv("HCU_STEP5_MEGAMOE_ALL_DECODE", str(int(all_rows)))
    monkeypatch.setenv("HCU_STEP5_PAIR32_MEGAMOE_DECODE", str(int(enabled)))
    fn = function(
        Path("vllm_hcu/forward_context_runtime.py"),
        "_step5_megamoe_dp_rows",
        {"os": os},
    )
    counts = SimpleNamespace(tolist=lambda: [rows, rows + 16])
    expected = (
        (rows, rows + 16)
        if enabled and rows in ((16, 32, 64, 128) if all_rows else (64,))
        else None
    )
    assert fn(rows, counts) == expected
    assert fn(rows, None) is None


@pytest.mark.parametrize(
    "counts,expected",
    [
        ((16, 16), True),
        ((32, 32), True),
        ((64, 64), True),
        ((128, 128), True),
        ((16, 32), False),
        ((0, 16), False),
        ((129, 129), False),
        (None, False),
        ((16,), False),
    ],
)
def test_all_ranks_make_the_same_selection(counts, expected):
    fn = function(
        Path("vllm_hcu/model_executor/layers/fused_moe/step5_pair32_megamoe.py"),
        "eligible_dp_rows",
        {"ALLOWED_DP_ROWS": (16, 32, 64, 128)},
    )
    assert fn(counts) == expected


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
    monkeypatch.setenv("HCU_STEP5_PAIR32_MEGAMOE_DECODE", "1")
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
