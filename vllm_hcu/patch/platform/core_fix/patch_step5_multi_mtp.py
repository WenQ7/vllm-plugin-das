# SPDX-License-Identifier: Apache-2.0
"""Preserve Step5 stage selection and FP32 feedback in the V2 MTP runner."""

from contextvars import ContextVar
from functools import wraps
from types import ModuleType

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator"
PATCH_ID = "platform.core_fix.step5_multi_mtp"
TARGETS = (
    f"{TARGET_MODULE}.MultiModuleMTPSpeculator",
    f"{TARGET_MODULE}.dispatch_cg_and_sync_dp",
)

# Scope the adjustment to this speculator's idle-rank proposal. The target's
# DP state is immutable and already agreed by all ranks; no new collective or
# change to the common dispatcher is needed.
_DUMMY_DP_SYNC = ContextVar("step5_mtp_dummy_dp_sync", default=None)


def apply_to_module(module: ModuleType) -> bool:
    if getattr(module, "_vllm_hcu_step5_multi_mtp", False):
        return False
    cls = module.MultiModuleMTPSpeculator
    original_init = cls.__init__
    original_run = cls._run_model
    original_propose = cls.propose
    original_dispatch = module.dispatch_cg_and_sync_dp

    @wraps(original_init)
    def init(self, vllm_config, device):
        original_init(self, vllm_config, device)
        draft = vllm_config.speculative_config.draft_model_config.hf_config
        if draft.model_type == "step5_mtp":
            # Each Step5 decoder returns an FP32 residual. Keep it until the
            # next module's normalization/projection rather than round early.
            self.hidden_states = self.hidden_states.float()
            self.cached_target_hidden_states = self.cached_target_hidden_states.float()

    @wraps(original_run)
    def run(self, num_tokens, attn_metadata, slot_mappings,
            num_tokens_across_dp, spec_module_idx, *args, **kwargs):
        if getattr(getattr(self.model, "config", None), "model_type", None) == "step5_mtp":
            # Set at the runner boundary even if the forward uses a graph.
            # Python stage selection is captured into each graph's head GEMM.
            self.model.set_mtp_step(spec_module_idx)
        return original_run(self, num_tokens, attn_metadata, slot_mappings,
                            num_tokens_across_dp, spec_module_idx, *args, **kwargs)

    @wraps(original_propose)
    def propose(
        self, input_batch, attn_metadata, slot_mappings, last_hidden_states,
        aux_hidden_states, num_sampled, num_rejected, last_sampled,
        next_prefill_tokens, temperature, seeds, dp_sync=None, dummy_run=False,
        skip_attn_for_dummy_run=False, mm_inputs=None, is_profile=False,
    ):
        is_step5 = (
            getattr(getattr(self.model, "config", None), "model_type", None)
            == "step5_mtp"
        )
        token = _DUMMY_DP_SYNC.set(dp_sync if is_step5 and dummy_run else None)
        try:
            return original_propose(
                self, input_batch, attn_metadata, slot_mappings,
                last_hidden_states, aux_hidden_states, num_sampled,
                num_rejected, last_sampled, next_prefill_tokens, temperature,
                seeds, dp_sync=dp_sync, dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                mm_inputs=mm_inputs, is_profile=is_profile,
            )
        finally:
            _DUMMY_DP_SYNC.reset(token)

    @wraps(original_dispatch)
    def dispatch(
        cudagraph_manager, num_reqs, num_tokens, uniform_token_count,
        dp_size, dp_rank, max_query_len=None, need_eager=False,
        num_active_loras=0, parallel_config=None, allow_ubatching=False,
        uniform_decode=False, dp_sync=None,
    ):
        if dp_sync is not None and dp_sync is _DUMMY_DP_SYNC.get():
            # PIECEWISE target dispatch pads tokens without padding requests.
            # InputBatch.make_dummy redistributes those padded tokens, so its
            # artificial query length can differ from the agreed decode length.
            # Dummy outputs are discarded; retain the target's global agreement.
            # The original dispatcher still checks padded tokens and graph mode.
            uniform_token_count = dp_sync.uniform_token_count
        return original_dispatch(
            cudagraph_manager, num_reqs, num_tokens, uniform_token_count,
            dp_size, dp_rank, max_query_len=max_query_len,
            need_eager=need_eager, num_active_loras=num_active_loras,
            parallel_config=parallel_config, allow_ubatching=allow_ubatching,
            uniform_decode=uniform_decode, dp_sync=dp_sync,
        )

    cls.__init__ = init
    cls._run_model = run
    cls.propose = propose
    module.dispatch_cg_and_sync_dp = dispatch
    module._vllm_hcu_step5_multi_mtp = True
    return True
