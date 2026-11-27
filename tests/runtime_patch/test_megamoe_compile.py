"""Collectives must not execute in independently scheduled compiler benchmarks."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

p=Path(__file__).resolve().parents[2]/'vllm_hcu/megamoe_compile.py'
spec=importlib.util.spec_from_file_location('_mega_compile_policy',p)
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)

class Compilation:
    def __init__(self,ops):self.splitting_ops=ops


def test_collective_boundaries_preserve_attention_and_are_idempotent():
    original=['vllm::unified_attention_with_output']
    cfg=NS(kernel_config=NS(moe_backend='deep_gemm_mega_moe'),
        model_config=NS(enforce_eager=False),compilation_config=Compilation(original))
    assert policy.configure_megamoe_splitting(cfg)
    assert original==['vllm::unified_attention_with_output',*policy.MEGAMOE_SPLITTING_OPS]
    assert not policy.configure_megamoe_splitting(cfg)


def test_other_backends_and_eager_keep_their_policy():
    cfg=NS(kernel_config=NS(moe_backend='triton'),compilation_config=Compilation(None))
    assert not policy.configure_megamoe_splitting(cfg)
    cfg.kernel_config.moe_backend='deep_gemm_mega_moe'
    cfg.model_config=NS(enforce_eager=True)
    assert not policy.configure_megamoe_splitting(cfg)
    cfg.model_config.enforce_eager=False
    assert not policy.configure_megamoe_splitting(cfg)
    assert cfg.compilation_config.splitting_ops is None
    cfg.compilation_config.splitting_ops=['vllm::unified_attention_with_output']
    assert policy.apply_bound_megamoe_splitting(cfg.compilation_config)
    assert cfg.compilation_config.splitting_ops==['vllm::unified_attention_with_output',*policy.MEGAMOE_SPLITTING_OPS]
