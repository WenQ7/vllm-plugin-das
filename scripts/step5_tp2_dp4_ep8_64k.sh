#!/usr/bin/env bash
# Launch the selected Step5 TP2/DP4/EP8 profiles with DCP disabled.
set -euo pipefail
stage="${1:-decode}"
sampling="${2:-standard}"
case "$stage" in prefill|decode) ;; *) echo "Usage: $0 [prefill|decode] [standard|synthetic]" >&2; exit 2;; esac
case "$sampling" in standard|synthetic) ;; *) echo "MTP sampling must be standard or synthetic" >&2; exit 2;; esac
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 HSA_FORCE_FINE_GRAIN_PCIE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_STEP4_DSA_FP8_MATH=off VLLM_STEP4_DISABLE_DSA=0
export VLLM_STEP5_DSA_TORCH_SELECTOR=1 VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=LBNHC VLLM_USE_AOT_COMPILE=0
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 VLLM_STEP5_LL_GEMM_TUNING=0
export VLLM_STEP5_SEQUENCE_PARALLEL_MOE=1 VLLM_STEP5_DSA_SHORT_QUERY=1
export VLLM_STEP5_DEVICE_REQUEST_INDICES=1 VLLM_STEP5_BF16_TILE_TUNING=1
export VLLM_STEP5_TP2_HT_BF16_CHUNKS=0 VLLM_STEP5_DSA_FUSED_SELECTOR=1
export VLLM_HCU_SHARED_EXPERTS_EARLY_LAUNCH=0 VLLM_HCU_SHARED_EXPERTS_STREAM_FORCE=0
export VLLM_SERVER_DEV_MODE=1 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
unset NCCL_TOPO_FILE VLLM_STEP5_HOST_REQUEST_INDICES
if [[ "$stage" == prefill ]]; then
  backend=deepep_high_throughput; tokens=6144; sequences=64; memory=0.89
  export VLLM_STEP5_HT_COMPACT_WORKSPACE=1 VLLM_STEP5_SP_LL_WORKSPACE=0 VLLM_STEP5_MTP_DRAFT_FULL_GRAPH=0
  compilation='{"cudagraph_mode":"NONE","cudagraph_copy_inputs":false,"mode":3}'
else
  backend=deepep_low_latency; tokens=512; sequences=32; memory=0.92
  export VLLM_STEP5_HT_COMPACT_WORKSPACE=0 VLLM_STEP5_SP_LL_WORKSPACE=1 VLLM_STEP5_MTP_DRAFT_FULL_GRAPH=1
  compilation='{"cudagraph_mode":"PIECEWISE","cudagraph_copy_inputs":true,"mode":3,"cudagraph_capture_sizes":[4,8,16,32,64,128,256,512]}'
fi
speculation='{"method":"mtp","num_speculative_tokens":3}'
if [[ "$sampling" == synthetic ]]; then
  speculation='{"method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"synthetic","synthetic_acceptance_rates":[0.838992,0.598322,0.362686]}'
fi
exec vllm serve "${STEP5_MODEL:-/data/model/step-5-preview-0913-fp8-per-channel}" \
  --tokenizer "${STEP5_TOKENIZER:-/workspace/step5-adaptation/tokenizer-bytelevel}" \
  --tensor-parallel-size 2 --data-parallel-size 4 --pipeline-parallel-size 1 --decode-context-parallel-size 1 --enable-expert-parallel \
  --dtype bfloat16 --moe-backend deep_gemm --all2all-backend "$backend" \
  --block-size 64 --max-model-len 69632 --max-num-seqs "$sequences" \
  --max-num-batched-tokens "$tokens" --gpu-memory-utilization "$memory" \
  --no-enable-prefix-caching --compilation-config "$compilation" \
  --speculative-config "$speculation" --override-generation-config '{"eos_token_id":[1,2,128007]}' \
  --host 127.0.0.1 --port "${STEP5_PORT:-18106}" --served-model-name step5 --api-server-count 1
