# SPDX-License-Identifier: Apache-2.0
"""Real-model PR checks using the installed v0.33 Core, Adapter and Plugin."""

import os
from pathlib import Path

import pytest
import yaml

from tests.integration.model_runtime import run_vllm_case
from tests.integration.server.evalscope_server import (
    create_evalscope_work_dir,
    run_evalscope_server_test,
)
from tools.run_v033_ci import ROOT, load_model_cases

pytestmark = [pytest.mark.hcu, pytest.mark.model]
MODEL_ROOTS = (
    "/public/opendas/DL_DATA/llm-models",
    "/public4/opendas/DL_DATA/llm-models",
)
GSM8K_PATHS = (
    "/public/opendas/DL_DATA/ci_datasets/gsm8k",
    "/public4/opendas/DL_DATA/ci_datasets/gsm8k",
)


def _cases(kind):
    return [
        pytest.param(case, id=case["id"], marks=[
            pytest.mark.hcu_count(case["cards"]),
            *([pytest.mark.multi_hcu] if case["cards"] > 1 else []),
        ])
        for case in load_model_cases() if case["kind"] == kind
    ]


def _model_resource(relative, metadata):
    roots = os.environ.get("HCU_CI_V033_MODEL_ROOTS")
    candidates = [Path(p) / relative for p in (roots.split(os.pathsep) if roots else MODEL_ROOTS)]
    for path in candidates:
        if (path / metadata).is_file():
            return path.resolve()
    pytest.fail(f"missing {metadata} for {relative}: checked {candidates}")


def _gsm8k_data_path():
    override = os.environ.get("HCU_CI_V033_GSM8K_PATH")
    candidates = [Path(p) for p in ([override] if override else GSM8K_PATHS)]
    for path in candidates:
        # EvalScope's fixed examples come from the train split.
        if all(next(path.rglob(split + "-*.parquet"), None)
               or next(path.rglob(split + ".parquet"), None) for split in ("train", "test")):
            return path.resolve()
    pytest.fail(f"GSM8K needs local train and test parquet data: checked {candidates}")


@pytest.mark.external_service("evalscope")
@pytest.mark.parametrize("case", _cases("gsm8k"))
def test_gsm8k(case, monkeypatch):
    model = _model_resource(case["model"], "config.json")
    dataset = _gsm8k_data_path()
    config = yaml.safe_load((ROOT / "tests/models" / case["config"]).read_text(encoding="utf-8"))
    config["model"] = str(model)
    server = config["server"]
    server.setdefault("environment", {}).update(
        VLLM_PLUGINS="hcu_adapter", HCU_ADAPTER_HOST="vllm",
    )
    if "quantization" in case:
        args = server["args"]
        args[args.index("--quantization") + 1] = case["quantization"]

    work_dir = create_evalscope_work_dir(Path(os.environ["HCU_V033_MODEL_ARTIFACTS"]) / "evalscope")
    monkeypatch.setenv("VLLM_HCU_EVAL_WORK_DIR", str(work_dir))
    monkeypatch.setenv("VLLM_HCU_GSM8K_DATASET", str(dataset))
    monkeypatch.delenv("VLLM_HCU_EVAL_LIMIT", raising=False)
    monkeypatch.delenv("HCU_V033_MODEL_PATH", raising=False)
    run_evalscope_server_test(
        config, model_env="HCU_V033_MODEL_PATH", model_label=case["id"],
        required_hcu_count=case["cards"],
    )


@pytest.mark.parametrize("case", _cases("lora"))
def test_lora(case):
    model = _model_resource(case["model"], "config.json")
    adapter_a, adapter_b = (
        _model_resource(relative, "adapter_config.json") for relative in case["adapters"]
    )
    result = run_vllm_case(
        "lora-switching", model, timeout_s=1800, vllm_plugins="hcu_adapter",
        extra_args=["--lora-a", str(adapter_a), "--lora-b", str(adapter_b)],
    )
    for key, expected_adapter in [
        ("base", None), ("adapter_a", "adapter-a"), ("adapter_b", "adapter-b"),
        ("adapter_a_again", "adapter-a"),
    ]:
        (record,) = result[key]
        assert record["lora_name"] == expected_adapter
        _assert_generation(record, max_tokens=8)
    assert result["adapter_a"][0]["token_ids"] == result["adapter_a_again"][0]["token_ids"]


def _assert_generation(record, *, max_tokens):
    assert record["prompt_token_count"] > 0
    assert 1 <= len(record["token_ids"]) <= max_tokens
    assert record["finish_reason"] in {"length", "stop", "eos"}


@pytest.mark.parametrize("case", _cases("tp_ep"))
def test_tp_ep(case):
    model = _model_resource(case["model"], "config.json")
    cards = case["cards"]
    result = run_vllm_case(
        "tp-ep-smoke", model, timeout_s=5400 if cards == 8 else 3600,
        vllm_plugins="hcu_adapter",
        extra_args=[
            "--tensor-parallel-size", str(cards),
            "--gpu-memory-utilization", str(case["gpu_memory_utilization"]),
        ],
        extra_env={"VLLM_HCU_TEST_TP_EP_QUANTIZATION": case["quantization"]}
        if "quantization" in case else None,
    )
    # Check the live engine config, not just what the launcher requested.
    parallel = result["parallel_config"]
    assert parallel["tensor_parallel_size"] == cards
    assert parallel["data_parallel_size"] == 1
    assert parallel["enable_expert_parallel"] is True
    assert parallel["world_size"] == cards
    assert len(result["output"]) == 2
    for record in result["output"]:
        _assert_generation(record, max_tokens=4)
