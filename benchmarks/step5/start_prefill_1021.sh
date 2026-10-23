#!/usr/bin/env bash
# Run inside the matching 1021 container. This script uses all four nodes.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
rank="${1:?Pass NODE_RANK: p5c1=0, p5c2=1, p5c3=2, p5c4=3}"
case "$rank" in 0|1|2|3) ;; *) printf 'NODE_RANK must be 0..3\n' >&2; exit 2 ;; esac
: "${STEP5_HEAD_IP:?Set the head communication IP}"
: "${VLLM_HOST_IP:?Set this node communication IP}"
: "${STEP5_TOKENIZER:?Set the external benchmark tokenizer directory}"
export STEP5_SOURCE_DIR="${STEP5_SOURCE_DIR:-$repo_root}"
export STEP5_MODEL_DIR="${STEP5_MODEL_DIR:-/data/model/Step5-Channel-FP8-w8a8}"
export STEP5_CACHE_DIR="${STEP5_CACHE_DIR:-/var/cache/xuwq_step5}"

# dense_1021: recorded baseline. g1_1021/g2_1021: explicitly selected candidates.
export STEP5_PREFILL_PROFILE="${2:-${STEP5_PREFILL_PROFILE:-dense_1021}}"
export STEP5_PREFILL_DP_SIZE=16
export STEP5_PREFILL_BATCHED_TOKENS=8192
export STEP5_PREFILL_MAX_MODEL_LEN=70000
export STEP5_PREFILL_GPU_MEMORY_UTILIZATION=0.875
export STEP5_PREFILL_MOE_BACKEND=auto STEP5_PREFILL_EP_FUSED_MOE=1
export STEP5_PREFILL_PLACEMENT="${STEP5_PREFILL_PLACEMENT:-/var/lib/xuwq_step5_assets/expert_placement_prefill_8192_ep16_20261020.pt}"
export STEP5_STATE_DIR="${STEP5_STATE_DIR:-/var/lib/xuwq_step5_1021/manual-prefill-1021}"
export STEP5_API_HOST="${STEP5_API_HOST:-127.0.0.1}"
export STEP5_PORT="${STEP5_PORT:-8018}"
exec bash "$repo_root/benchmarks/step5/launch_multinode.sh" prefill "$rank"
