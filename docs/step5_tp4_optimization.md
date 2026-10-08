# Step5 TP4 performance profile

Scope: one eight-card BW1101 host, TP4 + DP2 + EP8, PP1, DCP1.
Benchmark only concurrency 16/32, input 65536, Prefill output 1, Decode output 4096.
The token budget is 16384, maximum sequences 64, GPU memory utilization 0.85,
maximum model length 1048576, and prefix caching is disabled.

## Run

From the repository root:

    bash scripts/step5_tp4_8card.sh standard

The default uses ordinary MTP3 sampling. Performance comparison with fixed
synthetic 60% acceptance must explicitly select:

    bash scripts/step5_tp4_8card.sh synthetic

Synthetic acceptance uses rates [0.838992,0.598322,0.362686]; this is a performance
protocol, not a model accuracy result. The supplied benchmark uses token-ID
completions and one measured iteration per candidate.

The profile enables the qualified unified DeepEP Auto service: HT for Prefill
and mixed batches, LL for coordinated short Decode. Target and draft use
FULL_DECODE_ONLY graphs. PP and DCP are fixed at 1; DCP2 is rejected.

External dependencies on the current machine:

- Model: /data/model/step-5-preview-0913-fp8-per-channel
- Tokenizer: /workspace/step5-adaptation/tokenizer-bytelevel
- QKV calibration: /workspace/step5-attention-optimization/qkv-fp8-output-calibration-inputs-v1.pt
- rocBLAS library: /workspace/step5-tp4-optimization-20261008/rocblas-tuning-tp4-v1/auto_select_tools/optimization_configs/new/config/library_gpu6

Override STEP5_MODEL, STEP5_TOKENIZER, STEP5_QKV_CALIBRATION,
STEP5_TP4_ROCBLAS_LIBRARY and STEP5_PORT when those paths or the endpoint change.
MoE configuration comes from configs/step5_moe in this repository.
The launcher clears the inherited NCCL_TOPO_FILE; the container default describes
a different physical topology.

## Qualified same-service results

These measurements use the bounded mixed projections and small-M QKV path,
with shared auxiliary/split overlap and custom AllReduce disabled.

| Global BS | BS per DP | Prefill tok/s | Maximum TTFT s | Decode tok/s | Chunk ITL ms |
|---|---|---|---|---|---|
|16|8|11229.44|93.38|520.25|86.07|
|32|16|11231.23|186.72|872.60|102.97|

Prefill is a separate output 1 measurement. Prefill values observed during the
output 4096 Decode run are not substituted for it. ITL is the interval between
client output chunks; the average chunk contains about 2.8 tokens.

## Ported optimization coverage

| Area | TP4 behavior / evidence |
|---|---|
|MoE sequence parallel|SP4 with native FP8/BF16 experts, E44/N1536 configuration and gfx938 tile choices|
|DeepEP memory|Fresh HT routing counts; compact, bounded workspaces; shared HT/LL FP8 weights; 64-token per-dispatcher LL buffer|
|Attention/indexer|Native sparse paged Decode, Top512 visible selection bounds, QKnorm/RoPE packing, batched/grouped MTP indexer and persistent SWA metadata|
|CUDA graphs|Actual FULL target/draft Decode graphs; global DP coordination selects HT/LL consistently; mixed/Prefill remains outside graphs|
|GEMM selection|56 shapes numerically and actual-dispatch qualified; retain tuned library; three new Auto HT shapes have less than 0.3% component gain|
|QKV/mixed projection|Strict TP4 shape gates, small-M fused FP8 QKV, bounded merged attention/indexer/gate projections; large shapes retain original path|

The fused norm/FP8 trial had less than 1% whole-model gain and remains disabled.
Shared split dispatch had no component gain. Selective shared auxiliary stream
regressed whole-model Decode by 27.15%/24.48% at BS16/32 and remains experimental.
The installed DeepGEMM interface rejects native GEMM/combine overlap, and vLLM
rejects DBO with this MTP3 graph configuration. These are not enabled.

Native TP4 custom AllReduce passed 11 BF16/FP32 shapes and changed-input Graph
checks on both four-card groups. Whole-model and ordinary-sampling checks are
pending; STEP5_CUSTOM_ALL_REDUCE=on is an explicit experimental opt-in.
Component speedup alone is not adoption evidence.

A fixed 32-question ordinary MTP GSM8K subset scored 31/32 before and after the
combined QKV/mixed changes, with no truncation or preemption. Floating-point
output text differs. This is an integration subset, not full accuracy certification.

Current evidence root: /workspace/step5-tp4-optimization-20261008.
Key reports: bounded-mixed-model-performance-qualified-v1.json,
ordinary-sampling-paired-qualified-v1.json, tp2-to-tp4-coverage-audit-v10.json,
decode-trace-native-qualified-v1.json and custom-ar-native-qualified-v1.json.
The steady trace has 331474 GPU events across all 8 ranks in a 1212.198ms common
BS32 window. Cumulative overlapping kernel durations are not exclusive wall time.

The portable launcher was CPU-dry-run compared with both frozen launchers in
standard/synthetic modes and custom AR on/off. Arguments and relevant environment
match; the MoE config path changes to this repository's identical file.
