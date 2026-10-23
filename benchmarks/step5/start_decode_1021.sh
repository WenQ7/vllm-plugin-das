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

export STEP5_DECODE_PARALLEL=dp4tp4 STEP5_DECODE_PROFILE=itl6479
export STEP5_DECODE_GPU_MEMORY_UTILIZATION=0.85
export STEP5_DECODE_PLACEMENT="${STEP5_DECODE_PLACEMENT:-/var/lib/xuwq_step5_assets/expert_placement_decode_c64_ep16.pt}"
export STEP5_ROUTER_SHARED_MAX_M=64 STEP5_SHARED_STREAM_MIN_TOKENS=64
export STEP5_DECODE_EXPERT_GRID=1 STEP5_DECODE_FUSED_ROUTER_STREAM=0
unset STEP5_DECODE_NCCL_CHANNELS
export HCU_STEP5_DP_FABRIC=1 HCU_STEP5_CONTEXT_GRAPH_REGIONS=8736
export STEP5_STATE_DIR="${STEP5_STATE_DIR:-/var/lib/xuwq_step5_1021/manual-decode-1021}"
export STEP5_API_HOST="${STEP5_API_HOST:-0.0.0.0}"
export STEP5_PORT="${STEP5_PORT:-18118}"
exec bash "$repo_root/benchmarks/step5/launch_multinode.sh" decode "$rank"
