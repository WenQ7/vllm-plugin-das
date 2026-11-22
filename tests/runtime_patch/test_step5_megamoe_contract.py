"""CPU contracts for coordinated MegaMoE selection and layout propagation."""
from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[2]


def function(path, name, namespace):
    tree=ast.parse((ROOT/path).read_text(encoding="utf-8"))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),node],type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module,str(path),"exec"),namespace)
    return namespace[name]


@pytest.mark.parametrize("all_rows",[False,True])
@pytest.mark.parametrize("enabled",[False,True])
@pytest.mark.parametrize("rows",[0,16,32,64,128,129])
def test_context_preserves_coordinated_counts(monkeypatch,all_rows,enabled,rows):
    monkeypatch.setenv("HCU_STEP5_MEGAMOE_ALL_DECODE",str(int(all_rows)))
    monkeypatch.setenv("HCU_STEP5_PAIR32_MEGAMOE_DECODE",str(int(enabled)))
    fn=function(Path("vllm_hcu/forward_context_runtime.py"),"_step5_megamoe_dp_rows",{"os":os})
    counts=SimpleNamespace(tolist=lambda:[rows,rows+16])
    expected=(rows,rows+16) if enabled and rows in ((16,32,64,128) if all_rows else (64,)) else None
    assert fn(rows,counts)==expected
    assert fn(rows,None) is None


@pytest.mark.parametrize("counts,expected",[
    ((16,16),True),((32,32),True),((64,64),True),((128,128),True),
    ((16,32),False),((0,16),False),((129,129),False),(None,False),((16,),False)])
def test_all_ranks_make_the_same_selection(counts,expected):
    fn=function(Path("vllm_hcu/model_executor/layers/fused_moe/step5_pair32_megamoe.py"),
                "eligible_dp_rows",{"ALLOWED_DP_ROWS":(16,32,64,128)})
    assert fn(counts)==expected


@pytest.mark.parametrize("original",[False,True])
@pytest.mark.parametrize("mode",[None,"1001"])
def test_fallback_g1_and_g2_keep_the_weight_layout(monkeypatch,original,mode):
    fn=function(Path("vllm_hcu/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py"),
                "_step5_masked_gemm_options",{"os":os,"uses_original_pair32":lambda:original})
    for stage in ("G1","G2"):
        key=f"HCU_STEP5_DEEPGEMM_MASKED_{stage}_MODE"
        if mode is None:monkeypatch.delenv(key,raising=False)
        else:monkeypatch.setenv(key,mode)
        options=fn(stage)
        assert options.get("weight_layout")==("pair32" if original else None)
        assert options.get("config")==({"MODE":1001} if mode else None)
