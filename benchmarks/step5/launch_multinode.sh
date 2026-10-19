#!/usr/bin/env bash
# Four hosts x four BW1300_LC cards; benchmark-only synthetic MTP sampling.
set -euo pipefail

phase="${1:?Usage: launch_multinode.sh prefill|decode NODE_RANK}"
rank="${2:?NODE_RANK must be 0, 1, 2 or 3}"
case "$phase" in prefill|decode) ;; *) exit 2 ;; esac
case "$rank" in 0|1|2|3) ;; *) exit 2 ;; esac
: "${STEP5_HEAD_IP:?Set the head node IP reachable by all four nodes}"
: "${VLLM_HOST_IP:?Set the local node communication IP}"
: "${STEP5_TOKENIZER:?Set the tokenizer directory used by the benchmark}"
: "${STEP5_STATE_DIR:?Set a directory on the data-backed container disk}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_dir="${STEP5_SOURCE_DIR:-$repo_root}"
model_dir="${STEP5_MODEL_DIR:-/data/model/Step5-Channel-FP8-w8a8}"
export PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}"
export EP_FUSED_MOE_TABLE="${STEP5_EP_TABLE:-$source_dir/vllm_hcu/model_executor/layers/fused_moe/ep_table.json}"
export HOME="$STEP5_STATE_DIR/$phase/home"
export TMPDIR="$STEP5_STATE_DIR/tmp"
export XDG_CACHE_HOME="${STEP5_CACHE_DIR:-$STEP5_STATE_DIR/cache}"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export MIOPEN_USER_DB_PATH="$XDG_CACHE_HOME/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="$XDG_CACHE_HOME/miopen"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export VLLM_HOST_IP
export HSA_FORCE_FINE_GRAIN_PCIE=1 OMP_NUM_THREADS=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno2}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"
unset NCCL_TOPO_FILE NCCL_IB_DISABLE NCCL_MAX_NCHANNELS NCCL_MIN_NCHANNELS
unset VLLM_STEP5_EXPERT_PLACEMENT
unset VLLM_PLUGINS ROCBLAS_TENSILE_LIBPATH HIPBLASLT_TUNING_OVERRIDE_FILE
export GPU_MAX_HW_QUEUES=4
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=1
export VLLM_HCU_USE_FUSE_MOE_GATE=1
export VLLM_HCU_USE_FLASH_ATTN=1 VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_USE_PD_SPLIT=1 VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_HCU_USE_LIGHTOP_EP_SCATTER=1
export VLLM_HCU_USE_CUSTOM_SILU_AND_MUL=1
export VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM=1
export VLLM_ROCM_USE_AITER_MOE=1 VLLM_USE_FLASH_ATTN_FP8=1
export VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1 FA_DECODE_NUM_SPLITS=2
export VLLM_HCU_EP_FUSED_MOE=1 VLLM_HCU_EP_FUSED_COMBINE=1
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0
export VLLM_HCU_DP_CUSTOM_AG=1
export VLLM_HCU_STEP5_FP8_DENSE=1 VLLM_HCU_STEP5_FP8_BF16_MOE=1
export VLLM_HCU_STEP5_SWA_FUSED_QK=1 VLLM_HCU_STEP5_FUSED_GATE=1
export VLLM_HCU_TRITON_TOKEN_FP8_QUANT=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_HCU_PIN_MEMORY=1
export HIP_FORCE_DEV_KERNARG=1 VLLM_SERVER_DEV_MODE=1
export VLLM_USE_V2_MODEL_RUNNER=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_DEBUG=INFO
export VLLM_HCU_STEP5_QKV_INDEXER=1
export VLLM_STEP4_DSA_FLASH_ATTN=prefill,decode
export HCU_MTP_PREFILL_TAIL=1536 HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS=4096
export HCU_DP_MOE_PIPE_MIN_TOKENS=0 HCU_DBO_MIN_TOKENS=0 HCU_EP_DIRECT_SUM=0
export VLLM_STEP4_DSA_GRAPH_REGIONS=8704

if [[ "$phase" == prefill ]]; then
    tp=1; dp=16; local_dp=4; tokens=8192; max_len=70000; gpu_util=0.875
    export NCCL_MAX_NCHANNELS=32
else
    tp=4; dp=4; local_dp=1; tokens=32768; max_len=1048576; gpu_util=0.85
fi
extra=(--api-server-count 1)
if [[ "$rank" != 0 ]]; then
    extra=(--headless --data-parallel-start-rank "$((rank * local_dp))")
fi
command=(python3 -u -m vllm.entrypoints.cli.main serve "$model_dir"
    --tokenizer "$STEP5_TOKENIZER" --served-model-name step5
    --host 0.0.0.0 --port "${STEP5_PORT:-18118}"
    --tensor-parallel-size "$tp" --data-parallel-size "$dp"
    --data-parallel-size-local "$local_dp"
    --data-parallel-address "$STEP5_HEAD_IP"
    --data-parallel-rpc-port "${STEP5_RPC_PORT:-18192}"
    --distributed-executor-backend mp --enable-expert-parallel
    --all2all-backend allgather_reducescatter
    --max-model-len "$max_len" --max-num-seqs 64 --max-num-batched-tokens "$tokens"
    --kv-cache-dtype fp8_e4m3 --gpu-memory-utilization "$gpu_util"
    --no-enable-prefix-caching --max-cudagraph-capture-size 64
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["+swiglustep_and_mul"],"compile_sizes":[16,64]}'
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"synthetic","synthetic_acceptance_rates":[0.838992,0.598322,0.362686],"use_local_argmax_reduction":true}'
    --limit-mm-per-prompt '{"image":0}' "${extra[@]}")
if [[ "${STEP5_DRY_RUN:-0}" == 1 ]]; then
    printf 'EP_FUSED_MOE_TABLE=%q\n' "$EP_FUSED_MOE_TABLE"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
mkdir -p "$TMPDIR" "$HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" \
    "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$MIOPEN_USER_DB_PATH"
exec "${command[@]}"
