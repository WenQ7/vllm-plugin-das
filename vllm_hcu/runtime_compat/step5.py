# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Dependency-light Step5 family predicate shared by lazy adapters."""


def is_step5_config(config):
    model = getattr(config, "model_config", config)
    hf = getattr(model, "hf_config", model)
    text = getattr(hf, "text_config", hf)
    return getattr(text, "model_type", "") in ("step5", "step5_mtp") or any(
        name.startswith("Step5") for name in (getattr(hf, "architectures", None) or ())
    )


def is_step5_owner(owner=None):
    """Resolve ownership at construction time; unknown owners never opt in."""
    marker = getattr(owner, "_hcu_step5_owner", None)
    if marker is not None:
        return marker is True
    config = getattr(owner, "_hcu_vllm_config", None)
    if config is None:
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
    return config is not None and is_step5_config(config)


def is_step5_tp4_deepep_ll(config=None):
    """Identify the non-SP Step5 TP4/DP4 low-latency transport route."""
    if config is None:
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
    if config is None or not is_step5_config(config):
        return False
    parallel = getattr(config, "parallel_config", None)
    passes = getattr(getattr(config, "compilation_config", None), "pass_config", None)
    return bool(
        parallel is not None
        and getattr(parallel, "tensor_parallel_size", None) == 4
        and getattr(parallel, "data_parallel_size", None) == 4
        and getattr(parallel, "enable_expert_parallel", False)
        and not getattr(parallel, "enable_dbo", True)
        and getattr(parallel, "pipeline_parallel_size", None) == 1
        and getattr(parallel, "all2all_backend", None) == "deepep_low_latency"
        and not getattr(passes, "enable_sp", False)
    )


def configure_step5_deepep_hca(config, local_rank):
    """Bind rocSHMEM to the NIC selected by DeepEP for this logical device."""
    if not is_step5_tp4_deepep_ll(config):
        return
    import os

    mapping = os.environ.get("DEEP_EP_DEVICE_TO_HCA_MAPPING")
    if not mapping:
        return
    devices = {}
    for entry in mapping.split(","):
        device, hca = entry.split(":", 1)
        devices[int(device)] = hca.split(":", 1)[0]
    if local_rank not in devices or not devices[local_rank]:
        raise ValueError(f"DeepEP NIC mapping omits logical device {local_rank}")
    os.environ["ROCSHMEM_USE_IB_HCA"] = devices[local_rank]
