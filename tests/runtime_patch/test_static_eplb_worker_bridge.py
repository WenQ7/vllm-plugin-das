# SPDX-License-Identifier: Apache-2.0
"""Execute the small normal Worker/step bridges with CPU-only dependencies."""
import ast
from contextlib import nullcontext
import functools
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _worker_type():
    tree = ast.parse((ROOT / "vllm_hcu/v1/worker.py").read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HcuGPUWorker")
    methods = [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in {"load_model", "compile_or_warm_up_model", "step5_static_load_diag"}]
    assert len(methods) == 3

    class Parent:
        use_v2_model_runner = False

        def load_model(self, *, load_dummy_weights=False):
            assert self._hcu_static_load_diag_ready is False
            self.load_argument = load_dummy_weights
            if getattr(self, "fail_load", False):
                raise RuntimeError("load failure")

        def _compile_or_warm_up_model(self):
            assert self._hcu_static_load_diag_ready is False
            if getattr(self, "fail_compile", False):
                raise RuntimeError("compile failure")
            return self.result

    node = ast.ClassDef(name="Bridge", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=methods, decorator_list=[])
    if "type_params" in ast.ClassDef._fields:
        node.type_params = []
    namespace = dict(Parent=Parent, nullcontext=nullcontext,
                     runner_logger=SimpleNamespace(info=lambda *args: None))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "<actual Worker bridges>", "exec"), namespace)
    return namespace["Bridge"]


def _install_validation(monkeypatch):
    module = ModuleType("vllm_hcu.patch.worker")
    calls = []
    module.validate_worker_patches = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


def test_worker_arms_readiness_only_after_successful_load_and_compile(monkeypatch):
    validation = _install_validation(monkeypatch)
    worker = _worker_type()()
    assert not getattr(worker, "_hcu_static_load_diag_ready", False)
    worker.result = object()
    worker.load_model()
    assert not worker._hcu_static_load_diag_ready
    assert validation == [{"require_applied": True}]
    assert worker.compile_or_warm_up_model() is worker.result
    assert worker._hcu_static_load_diag_ready is True


@pytest.mark.parametrize("stage", ["load", "compile"])
def test_worker_failure_clears_stale_readiness(monkeypatch, stage):
    _install_validation(monkeypatch)
    worker = _worker_type()()
    worker._hcu_static_load_diag_ready = True
    setattr(worker, "fail_" + stage, True)
    with pytest.raises(RuntimeError, match=stage + " failure"):
        if stage == "load":
            worker.load_model()
        else:
            worker.compile_or_warm_up_model()
    assert worker._hcu_static_load_diag_ready is False


def test_dummy_weights_do_not_arm_worker_diagnostics(monkeypatch):
    _install_validation(monkeypatch)
    worker = _worker_type()()
    worker.result = object()
    worker.load_model(load_dummy_weights=True)
    assert worker.load_argument is True
    assert worker.compile_or_warm_up_model() is worker.result
    assert worker._hcu_static_load_diag_ready is False


def test_scope_exit_failure_does_not_arm_worker(monkeypatch):
    class Scope:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            raise RuntimeError("scope failure")

    module = ModuleType("vllm_hcu.v1.hcu_model_runner_v2")
    module.sampler_warmup_scope = Scope
    monkeypatch.setitem(sys.modules, module.__name__, module)
    worker = _worker_type()()
    worker.use_v2_model_runner = True
    worker.result = object()
    with pytest.raises(RuntimeError, match="scope failure"):
        worker.compile_or_warm_up_model()
    assert worker._hcu_static_load_diag_ready is False


def test_worker_rpc_preserves_json_string_arguments_without_arming(monkeypatch):
    module = ModuleType("vllm_hcu.distributed.static_eplb_diagnostics")
    calls, result = [], {"enabled": False}
    module.worker_rpc = lambda worker, **kwargs: calls.append((worker, kwargs)) or result
    monkeypatch.setitem(sys.modules, module.__name__, module)
    worker = _worker_type()()
    assert worker.step5_static_load_diag("status", "trial", "64") is result
    assert calls == [(worker, dict(action="status", session="trial", max_steps="64"))]
    assert not getattr(worker, "_hcu_static_load_diag_ready", False)


def _static_step():
    path = ROOT / "vllm_hcu/patch/worker/framework_opt/patch_offline_eplb.py"
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "step")

    def original(*args, **kwargs):
        raise AssertionError("Static diagnostics must not call the dynamic step")

    namespace = dict(os=os, functools=functools, original_step=original,
                     _paths=lambda config: ("/fixed/map.json", None))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["step"]


def test_static_step_default_does_not_import_or_run_diagnostics(monkeypatch):
    monkeypatch.delenv("HCU_STATIC_EPLB_LOAD_DIAGNOSTICS", raising=False)
    module = ModuleType("vllm_hcu.distributed.static_eplb_diagnostics")
    module.__getattr__ = lambda name: pytest.fail("Diagnostics imported by default: " + name)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert _static_step()(SimpleNamespace(parallel_config=object())) is None


@pytest.mark.parametrize("kwargs", [{}, {"is_dummy": True}, {"is_profile": True}])
def test_static_step_optin_forwards_only_phase_flags(monkeypatch, kwargs):
    monkeypatch.setenv("HCU_STATIC_EPLB_LOAD_DIAGNOSTICS", "1")
    module = ModuleType("vllm_hcu.distributed.static_eplb_diagnostics")
    calls = []
    module.on_static_step = lambda state, **flags: calls.append((state, flags))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    state = SimpleNamespace(parallel_config=object())
    assert _static_step()(state, **kwargs) is None
    assert calls == [(state, dict(is_dummy=kwargs.get("is_dummy", False), is_profile=kwargs.get("is_profile", False)))]
