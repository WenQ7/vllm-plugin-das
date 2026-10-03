# SPDX-License-Identifier: Apache-2.0
# Discard temporary Step5 DSA state when V2 replaces its KV allocation.

from functools import wraps
from types import ModuleType

TARGET_MODULE = "vllm.v1.worker.gpu.model_runner"
PATCH_ID = "platform.core_fix.step5_kv_cache"
TARGETS = (f"{TARGET_MODULE}.GPUModelRunner.initialize_kv_cache",)


def apply_to_module(module: ModuleType) -> bool:
    if getattr(module, "_vllm_hcu_step5_kv_cache", False):
        return False
    cls = module.GPUModelRunner
    original = cls.initialize_kv_cache

    @wraps(original)
    def initialize(self, *args, **kwargs):
        if getattr(self.model_config.hf_text_config, "model_type", None) == "step5":
            # The same model survives graph-memory profiling and real cache init.
            # Reset before builders can reuse a schedule from the previous pool.
            seen = set()
            for layer in self.model.modules():
                state = getattr(layer, "dsa_state", None)
                if state is not None and id(state) not in seen:
                    seen.add(id(state))
                    state.reset_cache()
        return original(self, *args, **kwargs)

    cls.initialize_kv_cache = initialize
    module._vllm_hcu_step5_kv_cache = True
    return True
