"""The requested DP16 decoder must retain its token budget and context limit."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def launcher(tmp_path):
    bash = shutil.which("bash")
    if not bash or subprocess.run([bash, "-c", "true"], capture_output=True).returncode:
        pytest.skip("A working Bash is required for launcher checks")
    env = {k: v for k, v in os.environ.items() if not k.startswith("STEP5_")}
    env.update(STEP5_HEAD_IP="172.20.2.22", VLLM_HOST_IP="172.20.2.22",
               STEP5_TOKENIZER="/external/tokenizer", STEP5_STATE_DIR=str(tmp_path),
               STEP5_DRY_RUN="1")

    def run(phase, rank=0, **settings):
        return subprocess.run([bash, str(ROOT / "benchmarks/step5/launch_multinode.sh"),
                               phase, str(rank)], env=dict(env, **settings),
                              capture_output=True, text=True, timeout=15)
    return run


@pytest.mark.parametrize("rank", range(4))
def test_dp16_decode_retains_fixed_limits_and_local_ranks(launcher, rank):
    result = launcher("decode", rank, STEP5_DECODE_PARALLEL="dp16tp1",
                      STEP5_DECODE_PROFILE="dp16_native")
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[-1])
    expected = {"--tensor-parallel-size": "1", "--data-parallel-size": "16",
                "--data-parallel-size-local": "4", "--max-num-batched-tokens": "32768",
                "--max-model-len": "1048576", "--long-prefill-token-threshold": "2048",
                "--gpu-memory-utilization": "0.91"}
    for key, value in expected.items():
        assert command[command.index(key) + 1] == value
    assert "VLLM_HCU_DP_PACKED_AG=0" in result.stdout
    assert "HCU_STEP5_BOUNDED_EP_WORKSPACE=1" in result.stdout
    assert "VLLM_STEP4_DSA_GRAPH_REGIONS=0" in result.stdout
    if rank:
        assert command[command.index("--data-parallel-start-rank") + 1] == str(rank * 4)
        assert "--headless" in command
    else:
        assert command[command.index("--api-server-count") + 1] == "1"


def test_existing_decode_and_prefill_defaults_stay_separate(launcher):
    for phase, tp, dp, budget, context in [
        ("decode", "4", "4", "32768", "1048576"),
        ("prefill", "1", "16", "8192", "70000"),
    ]:
        settings = {} if phase == "decode" else dict(
            STEP5_DECODE_PARALLEL="dp16tp1", STEP5_DECODE_PROFILE="dp16_native",
            STEP5_DECODE_GPU_MEMORY_UTILIZATION="0.91",
            STEP5_DECODE_LONG_PREFILL_TOKEN_THRESHOLD="2048")
        result = launcher(phase, **settings)
        assert result.returncode == 0, result.stderr
        command = shlex.split(result.stdout.splitlines()[-1])
        for key, value in {"--tensor-parallel-size": tp, "--data-parallel-size": dp,
                           "--max-num-batched-tokens": budget, "--max-model-len": context}.items():
            assert command[command.index(key) + 1] == value
        assert "--long-prefill-token-threshold" not in command


def test_prefill_can_screen_native_deepgemm_without_changing_recipe(launcher):
    baseline = launcher("prefill")
    candidate = launcher("prefill", STEP5_PREFILL_EP_FUSED_MOE="0")
    assert baseline.returncode == candidate.returncode == 0
    assert "VLLM_HCU_EP_FUSED_MOE=1" in baseline.stdout
    assert "VLLM_HCU_EP_FUSED_MOE=0" in candidate.stdout
    baseline_command = shlex.split(baseline.stdout.splitlines()[-1])
    candidate_command = shlex.split(candidate.stdout.splitlines()[-1])
    assert baseline_command == candidate_command


def test_invalid_prefill_expert_kernel_switch_fails(launcher):
    assert launcher("prefill", STEP5_PREFILL_EP_FUSED_MOE="deepgemm").returncode != 0


@pytest.mark.parametrize("settings", [
    {"STEP5_DECODE_PARALLEL": "dp8tp2"},
    {"STEP5_DECODE_PARALLEL": "dp16tp1", "STEP5_DECODE_PROFILE": "itl6479", "STEP5_DECODE_PLACEMENT": "/external/placement.pt"},
    {"STEP5_DECODE_PARALLEL": "dp16tp1", "STEP5_DECODE_LONG_PREFILL_TOKEN_THRESHOLD": "-1"},
])
def test_invalid_or_incompatible_decode_modes_fail_before_launch(launcher, settings):
    assert launcher("decode", **settings).returncode != 0
