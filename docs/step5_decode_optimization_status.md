# Step5 decode optimization status

This note records the production decisions for the single-node BW1101
`TP2 + DP4 + EP8 + MTP3` profile. Measurements use the per-channel FP8 model
and the same 64K decode topology unless a row explicitly says operator-only.

## Adopted

- The Triton MoE helper loads the checked-in `E=44,N=1536,device_name=BW1101.json`
  tuning file through `VLLM_TUNED_CONFIG_FOLDER`. It is not
  the tile template for masked FP8 DeepGEMM; that path separately chooses
  measured MODE=1101 for up to 512 global tokens. The measured improvement is small
  but repeatable: about 0.17% on short decode and 0.24% on the matching prefill
  comparison.
- Short Step5 sparse attention uses the FlashAttention-style sparse path for
  one to 128 target-verification tokens. At the production geometry,
  single-token decode improved from 0.0892 ms to 0.0391 ms (2.28x), and the
  16-token MTP shape improved from 0.1680 ms to 0.0910 ms (1.85x). A BS32
  service run improved decode from 689.32 to 696.27 token/s (about 1.0%). Long
  prefill remains on the tiled Triton path.
- The HD192 short-query kernel now uses one software-pipeline stage on BW1101.
  The previous two-stage launch reduced occupancy because a second K/V tile
  exceeded the useful LDS working set. At 64K/top-513 the per-layer Attention
  time improved from 0.3086 to 0.2910 ms at M=32 (1.06x), 0.6106 to 0.3395 ms
  at M=64 (1.80x), and 1.2131 to 0.4000 ms at M=128 (3.03x). M=1/4/16 and
  effective region counts 64/256/513 also improved, with bit-identical output
  and LSE in every comparison. The warmed total-BS32 synthetic service check
  was neutral within run variance (1596.26/528.64 before versus
  1586.73/525.49 total/output token/s after), as DP4 maps it to the smallest
  M=32 target shape; total BS64 maps to the larger M=64 shape.
- The stage-1 short-query kernel uses a BW1101-specific split schedule instead
  of stopping as soon as its grid fills the 64 CUs. At top-513 this reduces
  M=32 from 0.2910 to 0.1344 ms (2.17x), M=64 from 0.3394 to 0.2179 ms
  (1.56x), and M=128 from 0.4002 to 0.3693 ms (1.08x). Against the immediately
  preceding stage-1 service run, with identical synthetic MTP acceptance and
  per-position accepted-token counts, total/output throughput improved from
  1586.73/525.49 to 1598.64/529.43 token/s (+0.75%), mean TPOT fell from
  49.56 to 49.17 ms, and mean ITL fell from 135.35 to 134.26 ms.
- Split-state merge now writes directly into the Attention output buffer and
  omits the unused final LSE, eliminating one output copy per sparse layer.
  Including that copy, the isolated M=32/64/128 paths improve from
  0.1377/0.2218/0.3744 ms to 0.1342/0.2178/0.3681 ms, with bit-identical
  output. The split-state merge also selects one wave for two splits and eight
  waves for four splits, reducing the complete direct-output path further to
  0.1313/0.2134/0.3620 ms while preserving the reference reduction result.
- The formal `step5_tp2_dp4_ep8_64k.sh` profile currently keeps QKV in BF16:
  it does not set `VLLM_STEP5_QKV_FP8_W8A8`, whose model default is 0. The
  optional `step5_attention_8card.sh` profile enables output-MSE-calibrated
  QKV FP8 and requires a calibration file. That path uses LightOP's
  channel-wise FP8 hipBLASLt kernel, which was faster than the installed FP8
  alternatives. Those optional-kernel results are not evidence that the
  formal profile was running FP8 QKV. The 698.48 -> 806.07 BS64 comparison
  below retained BF16 QKV on both sides.
- MTP3 short-prefill indexer scoring is batched across requests and reads the
  summary pool directly through the block table. At 64K context, the per-layer
  score stage improved from 1.2004 ms to 0.3430 ms at total BS32 (3.50x), and
  from 2.3946 ms to 0.6778 ms at total BS64 (3.53x), with bit-identical fp32
  scores.
- The MTP3 scorer now puts the four consecutive queries of each request into
  one tensor-core M tile and reuses its summary K tile. The per-layer scorer at
  total BS32 improved from 0.3723 ms to 0.1145 ms (3.25x); at BS64 it improved
  from 0.7241 ms to 0.2096 ms (3.45x), with bit-identical scores including
  groups that cross a region boundary. With synthetic acceptance fixed at
  61.21%, the BS32 1024-to-512 service comparison improved total/output
  throughput from 1557.44/515.79 to 1596.26/528.64 token/s (2.49%), reduced
  mean TPOT from 50.24 to 48.74 ms, and reduced mean ITL from 141.51 to
  137.27 ms.
- Q/K indexer normalization and partial RoPE share one launch for up to 128
  target-verification tokens. The original joint kernel used four warps and was
  slower at 64 tokens; using one warp keeps the 256-wide reduction in one wave.
  It improves M=32 from 0.01314 to 0.00840 ms, M=64 from 0.01362 to 0.00909 ms
  (1.50x), and M=128 from 0.01686 to 0.01148 ms (1.47x), bit-identically.
  Larger prefill shapes retain the reference two-launch path because the fused
  reduction begins to diverge there.
- Prefill radix selection now packs physical regions and appends the current
  causal region in the same kernel. At 64K/top-512 this reduces the selection
  metadata stage from 0.1184 ms to 0.1128 ms at BS32 (5.0%), and from 0.1537 ms
  to 0.1470 ms at BS64 (4.5%). Short verification no longer allocates or writes
  logical IDs that its untiled Attention kernel does not consume; at 8192
  regions this reduces 32/64/128-row selection from
  0.1130/0.1135/0.1481 ms to 0.1058/0.1060/0.1393 ms, with identical packed
  slots and counts. Long tiled prefill still retains logical IDs.
  A warmed BS32 check after combining the later exact indexer, direct-output,
  merge-wave, and metadata changes kept identical MTP acceptance but measured
  1593.50/527.73 total/output token/s versus 1598.64/529.43 immediately before
  them (-0.32%), so no additional BS32 service-level gain is claimed.

### Live BF16 expert tiles and ragged MTP batches

The three routed-expert layers that remain BF16 now use a device-counted
expert grid for the TP2/DP4/EP8 low-latency profile. The previous native grid
launched M tiles over the full 2048-row dispatch capacity for all 44 local
experts, even though most decode experts only received a few live rows.
The replacement launches expert/N tiles and loops over live M tiles using
GPU counts. It reuses the native BF16 tile calculation and the selected
BLOCK_SIZE_M/N/K, including its K-reduction order. It also works inside
PIECEWISE capture when the batch descriptor is absent.

With the production M128/N128/K64 configuration and 44 experts, the two
isolated GEMMs measured 7.7919 -> 1.9930 ms and 3.8295 -> 1.0122 ms.
Results and untouched padding were bit-identical, including graph replays
with zero, partial, and multiple M tiles. With M32/N64/K64 the same grid
change measured 6.8317 -> 1.2504 ms and 3.6132 -> 0.6479 ms; that tile
configuration is not newly selected by this change.

The warmed BS64, input1024/output512 synthetic-MTP service comparison
completed 64/64 requests. Output throughput improved 698.4773 -> 806.0666
token/s (+15.40%), mean TPOT 67.7000 -> 57.8990 ms (-14.48%), and mean
ITL 189.4302 -> 162.0062 ms (-14.48%). Both runs had exactly 11624 drafts,
34872 draft tokens, 21156 accepted tokens, and 60.6675843% acceptance.
Baseline result: `/tmp/step5_norm_baseline_bench/openai-chat-infqps-concurrency64-step5-20261006-223408.json`.
Candidate result: `/tmp/step5_bf16_compact_bench/openai-chat-infqps-concurrency64-step5-20261007-001232.json`.
This is a short-context service screening result; it does not establish a
new 64K/4K throughput figure. The formal Decode launcher enables
`VLLM_STEP5_BF16_COMPACT_GRID=1`; an explicit value of 0 restores the prior
path. The FP8 experts and extra shared-expert streams are unchanged.

Ragged MTP batches of 65--128 tokens now also use short-query DSA when
host metadata says every request has at most four queries. They no longer
need to satisfy the uniform four-query graph schedule to avoid CPU tiles.
Ordinary 65--128-token prefills keep the tiled path. Selector and Attention
routing share one predicate, so selection omits logical IDs exactly when
Attention will not consume them. Boundary/selection checks and 65/96/128
query numerical comparisons passed. The DSA checks passed 33 tests
(excluding an existing direct-import duplicate-registry test); the BF16
expert checks passed two tests with dynamic graph counts and strided output.

### Decode profiling scope

The original `/tmp/step5_profile` trace was collected from the beginning of
a benchmark and predominantly covered 512-token prefill chunks. The run
also failed during profiler teardown, so its per-layer scalar-read counts
must not be interpreted as steady Decode synchronization.

A replacement `/tmp/step5_decode_profile` trace delays capture by 100 worker
iterations and records three steady iterations on all eight ranks. Rank0
Attention inputs are [64,32,192]; sliding layers use direct paged Attention,
not a paged-to-contiguous prefill conversion. No scalar reads occur inside
Attention or indexer scopes. The 45 CPU scalar operations and 12 DtoH events
are outside those scopes. BF16 expert GEMMs took 86.869 ms of 442.601 ms
GPU execution across the three sampled steps (~19.6%); DeepEP combine took
53.893 ms (~12.2%). This evidence directed the live-tile optimization above.
The profiled run was stopped after traces were collected and is not used
for service throughput or correctness claims.

## Implemented candidates pending long-context adoption

### Full target-model graph

The target model can now capture all six uniform MTP3 graph sizes. Its
four-query DSA schedule keeps fixed addresses for compression, rollback
prefixes, request indices, and score storage, and is bit-identical to the
general scheduler. A 32-request drain test with staggered 16--78 token outputs
completed 32/32 requests without a VM fault or worker failure.

An earlier short-drain service A/B did not justify enabling it: the same synthetic workload
took 9.56 seconds with `FULL_AND_PIECEWISE` and 6.67 seconds with `PIECEWISE`, a
43.3% regression. The completion-drain interval was essentially unchanged
(2.92 versus 2.94 seconds), so that workload did not establish a steady
decode benefit and showed additional initial latency. The formal launcher therefore retains target
`PIECEWISE`; the MTP draft keeps its verified full decode graph.

After the live BF16 grid and ragged-query fixes, a matching warmed BS64,
input1024/output512 synthetic run measured 806.0666 -> 817.3194 output token/s
(+1.40%) and 57.8990 -> 56.9690 ms mean TPOT (-1.61%) when only the target
Graph mode changed. Both runs completed 64 requests with exactly 11624 drafts,
34872 draft tokens, and 21156 accepted tokens (60.6675843%). Real dispatcher
metrics recorded FULL target execution, including padded shrinking batches;
a fresh 32-request 16--78-token drain also completed every requested length.
Result: `/tmp/step5_compact_full_graph_bench/openai-chat-infqps-concurrency64-step5-20261007-002440.json`.
This single screening run shows a modest gain, not statistical significance or
64K/4K performance. FULL at memory utilization 0.92 initialized 924376 tokens
per DP (13.28 requests at 69632 tokens); BS64/DP4 requires 1114112 per DP.
The candidate is therefore pending long-context capacity/performance checks.

## Rejected after measurement

### Small-M QKV alternatives

For `K=4096,N=6912`, including per-token activation quantization:

| M | LightOP end-to-end | AITER Triton stock best | Result |
|---:|---:|---:|---|
| 1 | 0.0680 ms | 0.1033 ms | AITER 52% slower |
| 4 | 0.0676 ms | 0.0998 ms | AITER 48% slower |
| 16 | 0.0679 ms | 0.0998 ms | AITER 47% slower |
| 32 | 0.0707 ms | 0.0998 ms | AITER 41% slower |

A second M=1 search tried 24 small-M AITER tile variants. Its best GEMM was
0.0815 ms versus LightOP's 0.0610 ms, still 34% slower before activation
quantization. AITER has no `BW200B/BW1101-GEMM-A8W8` tuning file in the
installed package. LightOP W8A16 was also slower, so retaining an extra BF16
weight copy for a shape switch would add memory without a decode gain.

A Step5-specific FP32 Gemma norm + BF16 store + per-token FP8 quantization
prototype was also measured. At the production BS64 per-DP shape (M=64), it
was bit-identical to native Inductor norm plus LightOP quantization across 100
random inputs and reduced that isolated pair from 0.01377 to 0.00997 ms. The
end-to-end concurrency-64 1024-to-512 synthetic-MTP comparison nevertheless
regressed: baseline/fused output throughput was 698.48/695.17 token/s and mean
TPOT was 67.70/68.10 ms. Acceptance statistics were identical (60.6676%,
21,156 accepted of 34,872 drafted tokens). The extra custom-op boundary erased
the isolated gain, so the candidate was removed and is not enabled.

### Projection and shared-expert fusion

- The TP2 merged projection service comparison regressed decode from the
  225.46 token/s QKV-FP8 reference to 220.26 token/s. The mixed FP8/BF16
  projection kernel was also 15.5% slower than the two native GEMMs at the
  TP2 four-token sparse shape. These paths remain disabled.
- Merging only the four indexer projections reduced their M=32 operator time
  from 0.0407 ms to 0.0244 ms, but changed BF16 GEMM rounding from M=8 upward.
  In the identical 1024-to-512 BS32 service check, total throughput fell from
  1053.79 to 793.94 token/s and MTP acceptance fell from 29.82% to 23.38%.
  LightOP grouped BF16 GEMM cannot replace it because the installed gfx938
  wheel lacks its grouped-GEMM HSACO. AITER grouped GEMM requires N divisible
  by 128 (the gate projection has N=8); grouping the other three projections
  was about 0.66--0.77 ms versus 0.036--0.040 ms for native separate GEMMs.
- Fusing the attention head sigmoid gate into the output projection was also
  rejected. The prototype took 0.100--0.117 ms for M=4--64, more than twice
  the 0.046--0.062 ms native gate-plus-GEMM path, and was not bit-exact.
- Folding the indexer Q e4m3 rounding into fused QK-norm/RoPE was
  bit-identical and reduced the isolated norm-plus-scorer pipeline by 9.5% at
  BS32. It did not improve the fixed-acceptance service run: total/output
  throughput changed from 1596.26/528.64 to 1571.71/520.52 token/s and mean
  TPOT increased from 48.74 to 49.71 ms, so the standalone rounding pass stays.
- The fused main QK-norm/RoPE candidate was compared with the deployed
  Inductor path rather than only with eager PyTorch. Inductor carries FP32
  normalized values through RoPE before the BF16 store, so eager PyTorch
  differs by as much as 0.0625. The custom kernel matched compiled output
  exactly at M=1 and M=32; across the other tested M=4/16/64 shapes only one to
  three Q elements differed, by at most 0.0078125, and K was exact. It still
  adds a custom-op boundary around work Inductor already fuses. The earlier
  service run fell from 1053.79/348.99 to 926.54/306.85 total/output token/s,
  so the production profile retains the compiled native QK-norm/RoPE sequence.
- Early shared-expert launch on an auxiliary stream increased the 64K warm
  prefill maximum from 153.7 s to 324.4 s and reduced the observed active
  generation rate to 22.96 token/s. The existing token threshold does not
  remove contention at the production decode shape, so the formal profile
  keeps the shared-expert stream disabled. vLLM also rejects dual-batch
  overlap when speculative decoding or CUDA graphs are enabled, so DBO cannot
  overlap DeepEP in the required MTP3 graph-enabled profile.

## Constraints that cannot be enabled by a flag

### TP2 + DP4 + DCP2

This topology is invalid for Step5's GQA geometry. The model has four total KV
heads, and vLLM requires `tensor_parallel_size > num_kv_heads` before DCP may
split repeated KV heads. With TP2, startup rejects DCP2 before workers launch.
On an eight-card node the smallest valid DCP topology is TP8 + DP1 + DCP2,
which loses DP4 and is a different performance target. The formal TP2 launcher
fails early with this explanation when `STEP5_DCP_SIZE` is not 1.

## LightOP and AITER kernel audit

The installed LightOP and AITER packages were checked against the actual TP2
Step5 geometry: 32 local Q heads, two local KV heads, head dimension 192, and
`K=4096,N=6912` for QKV. AITER `fused_qk_rmsnorm_rope` and LightOP
`split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant` both reject this
geometry because their reviewed paths require head dimension 128. LightOP's
sparse-attention package exports index transforms and top-k helpers, but no
sparse attention compute kernel. AITER's sparse MLA kernel requires its MLA
cache layout and power-of-two dimensions, which do not match Step5's separate
GQA K/V cache and 192-dimensional heads.

For QKV, LightOP's small-M smooth-GEMM table has no tuned
`N=6912,K=4096` entry. Its public channel-wise hipBLASLt path is the optional
FP8-QKV backend and remains faster than the installed AITER alternatives in
the measurements above. Therefore none of the installed library kernels can
replace the remaining Step5-specific decode stages without changing semantics
or regressing the measured shape.

## Remaining kernel work

Short MTP queries now batch scoring across requests, and the BS32 Q/K
normalization shares one launch. Projection, summary update, radix selection,
and attention are still separate stages. LightOP and AITER only provide
MLA-specific fused kernels for incompatible head dimensions and cache
semantics. Fusing the remaining stages therefore requires a new Step5-specific
kernel rather than wiring an existing library operator into this backend.


### Grouped provider/head scoring candidate

`VLLM_STEP5_GROUPED_HEAD_INDEXER=1` folds provider groups and all index heads
into one short-query GEMM, then preserves the original sequential FP32
head accumulation. It retains the existing e4m3 activation rounding and
native causal/region masks. The actual TP2 geometry is G=2,H=4,D=256;
the initial G=1,H=8 prototype is not a production-geometry measurement.
Twenty-four native-reference tests passed with BF16 and production FP8 summary pools for queries/request 1--4, group-major
strided outputs, page-shuffled summaries, and graph replay with changing
positions, Q and weights. The feature remains opt-in pending a service check.
Operator measurements including activation rounding are in
`/tmp/step5_grouped_head_actual_geometry_bench.log` (BF16 pool) and
`/tmp/step5_grouped_head_fp8_summary_bench.log` (production FP8 pool).
The first service startup caught an FP8 masked-load fill type that BF16-only
tests missed. The fill is now floating-point zero; the added FP8-pool graph
tests cover it. That failed startup produced no benchmark result.

The 64K full-graph capacity candidate at memory utilization 0.97 was rejected
by startup's free-memory check. Lowering dispatch capacity from 2048 to 1024
by using a 256-token scheduler and smaller capture sizes worked, but 0.955
still ran out of memory during Graph warmup. Neither candidate reached serving
state or produced a performance result. Formal memory remains 0.92.

Production FP8-pool scoring (including activation rounding) selected BK128,
four warps and two stages. At 16 requests per DP, queries/request 1/2/3/4
measured 0.20251/0.20279/0.20233/0.20023 ms on native and
0.09926/0.10124/0.12289/0.12306 ms on grouped-head scoring. Every tested tile
variant produced bit-identical scores. BK64 was slower than native for q=1/2,
so the BF16-pool tile choice is not used for the production FP8 summary pool.
These are operator results; service gains are still pending.

The warmed BS64 input1024/output512 service screen with grouped-head scoring
completed 64/64 requests, output throughput 817.3194 -> 818.8987 token/s
(+0.19%), mean TPOT 56.9690 -> 56.8968 ms (-0.13%). Synthetic acceptance was
identical: 11624 drafts, 34872 draft tokens, 21156 accepted tokens. Both sides
used the full target Graph and BF16 QKV. The operator's long-context gain did
not become a material short-context service gain. The selected BK128 tests
passed 24 cases again after service startup.
Result: `/tmp/step5_grouped_head_v2_bench/openai-chat-infqps-concurrency64-step5-20261007-010820.json`.
The completed BS32 65536-input/4096-output screen is recorded below.

A native KV allocator cost audit also rejected treating the 20 padding-layer
warning as an automatic 27.78% capacity gain. Full and sliding groups share a
physical page pool. With 23 full and 72 sliding layers, regrouping from size23
to size24 reduces groups from five to four but increases the 69632-token
per-request bound from 2622750720 to 2694316032 bytes in the native calculator
(+2.73%). Smaller exact groups add substantial scheduler/group overhead.
No speculative KV regrouping change was adopted. Audit:
`/tmp/step5_kv_group_cost_audit.py`, `/tmp/step5_kv_group_cost_audit.log`.

The first BS32 long-context wrapper incorrectly imported the metadata adapter
before setting its DP argument, so the adapter retained DP=1 and capacity
preflight skipped the case. No measured 64K iteration ran in that attempt.
The corrected wrapper explicitly sets DP=4, retains the 0.9 guard, reads all
four startup capacities, and completed one BS32 iteration:
`/tmp/step5_grouped_head_graph64k_bs32_v2_benchmark.py`.
Result and per-iteration sidecar:
`/tmp/step5_grouped_head_graph64k_bs32_v2_result.json` and
`/tmp/step5_grouped_head_graph64k_bs32_v2_result.iterations.json`.
The manifest also records preemption counters, so queued/preempted runs cannot
be passed off as an unpreempted Decode comparison.


### Completed 64K full-target-graph screen

The grouped-head FP8-summary scorer and full target Decode Graph service
completed 32/32 requests with 65536 input tokens and 4096 output tokens each.
The valid common Decode window measured 859.36049 output token/s and mean
ITL 101.11884 ms (p50 100.98841, p99 104.18955). Window coverage was 35.7349%,
with 46827 output tokens over 54.49052 seconds. All four DP preemption
counter deltas were zero. The request token hashes match the original
random.Random(42 + request_index) input generator. QKV remained BF16.
This is one measured iteration using the fixed synthetic MTP acceptance
rates [0.838992, 0.598322, 0.362686], not an accuracy or natural-acceptance run.

The historical matching BS32 64K/4K screen measured 575.90817 token/s and
mean ITL 154.90923 ms, so cumulative throughput improved about 49.2% and ITL
fell about 34.7%. Several kernels, target Graph, and all-reduce configuration
changed; the historical run also had a separate long-context warmup while
this run had a short-context warmup. This is not a sole-variable indexer A/B.
The mixed prefill stage's 3214.38 token/s must not be reported as the
standalone input64K/output1 Prefill benchmark.

Evidence:
- /tmp/step5_grouped_head_graph64k_bs32_v2_result.json
- /tmp/step5_grouped_head_graph64k_bs32_v2_result.iterations.json
- /tmp/step5_grouped_head_graph64k_bs32_v2_manifest.json
- /workspace/step5-tp2-performance/tp2dp4-final-decode-synthetic-v4-final-single-round-v4-summary.json

### Small live-batch tile selection

The compact BF16 grid used to return before the existing small-batch tile
selection, leaving M128/N128 even for small Decode batches. It now selects
M32/N64/K64 for at most 512 global scheduler tokens, using already-present
CPU DP counts cached once per forward. Capture falls back to its padded
batch descriptor when dummy CPU counts are zero. It never copies expert
counts or DP device counts back to the CPU.

At E44/capacity2048, representative Decode counts changed compact up/down
GEMM from 1.97433/0.99295 ms to 1.16736/0.58911 ms. Extreme one-expert
skew prefers M128; this host-token policy does not claim to optimize skew.
All five tested tile variants preserved reference output and padding
bit-exactly. Six native-reference GPU tests include dynamic-count Graph
replays; thirteen CPU tests cover host-count caching and both grid paths.
The scorer also skips wholly invisible score tiles on-device, avoiding
full-width summary work in short-context FULL captures. Its 24 native
reference tests cover changing positions and empty regions under replay.
Combined kernel/policy tests passed 42 cases, followed by the added zero
dummy-count capture regression passing in the 13-case CPU policy run.

Evidence: /tmp/step5_live_bf16_tiles_bench.json,
 /tmp/step5_small_tiles_and_indexer_tests.log,
 /tmp/step5_bf16_tile_policy_capture_tests.log.
The running service predates these tile and causal-skip changes; the next
short service screen is needed before attributing an end-to-end gain.

The formal Decode launcher now defaults to FULL_AND_PIECEWISE for the
target model and enables grouped-head indexer scoring after the successful
64K BS32 screen. STEP5_TARGET_GRAPH_MODE=PIECEWISE and
VLLM_STEP5_GROUPED_HEAD_INDEXER=0 remain explicit fallback overrides.
Prefill retains Graph NONE. This does not establish 64K BS64 capacity.

The single warmed BS64 input1024/output512 service screen completed 64/64:
output throughput 818.89871 -> 827.83147 token/s (+1.09%), mean TPOT
56.89679 -> 55.90127 ms (-1.75%). Total input/output tokens were identical
(66176/32768), as were synthetic MTP counts (11624 drafts, 34872 draft
tokens, 21156 accepted tokens). No DP preemption counter increased.
Both sides retained full target Graph, grouped-head indexer and BF16 QKV;
the new side adds small live-batch tiles and invisible-score-tile skipping.
This is a one-round short-context screen, not a new 64K/4K measurement.
Result: /tmp/step5_small_tiles_v1_bench/openai-chat-infqps-concurrency64-step5-20261007-014602.json
Manifest: /tmp/step5_small_tiles_v1_manifest.json
Comparison: /tmp/step5_small_tiles_v1_comparison.json

### Rounding-fusion candidates rejected for the 64K target

Fusing BF16-to-e4m3-to-BF16 Q rounding into the grouped scorer removes one
launch and a temporary tensor. It passed the sampled bit-exact score and
Graph tests but repeated conversion in every key CTA regressed width8192
from 0.1012/0.1023/0.1221/0.1179 ms to
0.1088/0.1108/0.1483/0.1438 ms for queries/request 1/2/3/4.
The width128 benefit is not a reason to adopt it for the 64K workload.

Direct FP8 MMA reduced the width8192 q1/2/3 screen to
0.0548/0.0563/0.0947 ms but q4 differed from the native score by
3.814697265625e-6. It is not qualified for replacing the native scorer.
An interleaved-zero FP8-K prototype preserved the sampled native rounding
at every tested q count and Graph replay but slowed width8192 to
0.1354/0.1343/0.2142/0.1726 ms. Neither FP8-MMA prototype is in
production code or enabled in the service.

That completed BF16 service screen used PID3162510, source dd54506. Its Graph dispatch table explicitly shows FULL at real token
counts 12--64, including padded batches. The MoE log confirms loading
E=44,N=1536,device_name=BW1101.json and masked DeepGEMM mode1101.
The kernel and service evidence is preserved outside /tmp at
/workspace/step5-tp2-performance/live-tile-and-index-rounding-20261007.
The previous long-context evidence is preserved at
/workspace/step5-tp2-performance/grouped-head-full-graph-64k-bs32-v2.

### Small-M QKV and calibrated full-Graph FP8 screen

A new fused token-amax / activation-quant / GEMM / scale prototype matched
LightOp quantized bytes and per-token scales, including the zero-row scale
contract. All tested GEMM tiles were slower than the current hipBLASLt path
and differed in a small number of BF16 output elements. At M=1/4/16 the best
candidate was about 0.0488/0.0487/0.0494 ms versus 0.0349/0.0349/0.0345 ms
on native quant-plus-GEMM. It was not added to production code.

The installed LightOp small-M smooth config has no N6912/K4096 entry.
Six neighboring tile configs were tested through its raw smooth operator
at M=1/4/16; all failed the FP8 output reference and were slower than the
qualified channel-wise hipBLASLt path. They are not an FP8 replacement.

Graph-level precision microbenchmarks used actual TP2 K4096/N6912 and
included activation quantization. Native BF16 vs FP8 at M=1/4/16/32/64/128
measured 0.0498/0.0431/0.0432/0.0439/0.0417/0.0538 ms versus
0.0286/0.0286/0.0283/0.0315/0.0306/0.0413 ms. FP8 also won M256--4096.
These random-weight operator results do not justify switching small M
to BF16, and are not a model-accuracy evaluation.

The current profile plus output-MSE-calibrated QKV FP8 completed a single
warmed BS64 input1024/output512 screen: output throughput
827.83147 -> 838.88914 token/s (+1.34%), mean TPOT
55.90127 -> 55.18102 ms (-1.29%). All 64 requests completed with zero failures,
input/output token totals 66176/32768 and identical synthetic MTP counts
(11624 drafts, 34872 draft tokens, 21156 accepted tokens). No DP preemption
counter increased. Main target FULL Graph remained active.
Per-DP KV capacity rose from about 924201 to 994709 tokens (+7.63%).
This still does not fit 16 requests/DP at input65536/output4096.

The calibration file has all 92 main and three MTP groups and the matching
model-index hash. It uses independent document activations from DP0/1;
heldout DP2/3 and evaluation answers were excluded. That completed candidate
service used PID3217802, source1138074, with QKV FP8. Formal QKV stays BF16:
ordinary MTP accuracy and a 64K/4K FP8 comparison remain unverified.
Result: /tmp/step5_qkv_fp8_graph_v1_bench/openai-chat-infqps-concurrency64-step5-20261007-021335.json
Evidence: /workspace/step5-tp2-performance/qkv-small-m-and-decode-overlap-20261007.

### Decode-only shared-expert overlap candidate

STEP5_SHARED_EXPERT_OVERLAP=decode now enables an opt-in profile whose
shared-expert aux stream is gated by the scheduler's host BatchDescriptor:
uniform batches with 1--4 queries/request are eligible. The initial
candidate made missing, mixed, nonuniform or large chunks serial, including
the MK-internal fallback. The follow-up repair below preserves native MK
overlap while restricting only the added aux-stream schedule. The decision is cached once per ForwardContext without reading
a GPU count. This avoids relying only on the padded hidden-state shape,
which allowed the previous overlap candidate to compete with Prefill.
The formal default stays off and the standalone Prefill profile rejects
this option.

Fifteen tests passed, including the existing input-alias preservation
policy and native auxiliary-stream/Graph replays with changing BF16 inputs.
The tests cover both serial and overlapped branches and their output
dependencies. CLI checks verify default-off, Decode opt-in and rejection
of standalone Prefill/invalid modes. A full-model service performance
screen is still required before adopting this overlap profile.

The native MRv2 FULL capture caller was inspected before accepting the
overlap performance run: it sets runtime NONE with no BatchDescriptor.
The initial descriptor-only guard would therefore capture serial execution.
That v1 startup was explicitly superseded before a measured benchmark.
It was not restarted because of an observation timeout.

The guard now falls back to existing CPU query-start mirrors during FULL
capture, caches the result once per ForwardContext, and rejects large,
ragged, inconsistent or nonuniform query groups. No device tensor is copied
or reduced on the host. Twenty-eight tests passed after this repair,
including auxiliary-stream Graph replays without a descriptor, GPU-count
poison tests and mixed attention groups. The successful earlier 15-test
descriptor-only screen did not cover the actual FULL-capture call shape.
The repaired v2 service screen completed 64/64 at input1024/output512,
BS64, with the same input/output tokens and synthetic acceptance counts as
the preceding calibrated-FP8, no-overlap screen. Output throughput fell
838.88914 -> 706.43915 token/s (-15.79%), while mean TPOT rose
55.18102 -> 67.65255 ms (+22.60%). No DP preemption counter increased.
The opt-in profile is rejected; the formal default remains off.

The service env flags requested aux-stream overlap, but the captured log
contains no mode-selection diagnostic. Isolated stream/Graph tests passed;
the actual full-model aux-stream activation is not independently proven.
Therefore this result rejects the configured candidate, without attributing
the entire regression to verified communication/computation overlap.
Results and comparison are preserved in the evidence directory above.

Checkpoint shape inspection confirms Q/K/V rows 12288/768/768 at both SWA
and full Attention, including MTP: TP2 QKV width is N6912 in either case.
A separate N3840 SWA hypothesis was incorrect and is not used as a basis
for precision selection. The formal QKV precision remains BF16.


### Fold indexer E4M3 query rounding into short Q/K norm + RoPE

The indexer requires BF16 -> E4M3 -> BF16 activation rounding before its
score GEMM. Moving that stage into the grouped scorer regressed long
contexts because every key CTA repeated the conversion. The new candidate
instead rounds Q exactly once while the joint norm/RoPE kernel stores it.
It preserves the BF16 store boundary before E4M3 conversion and keeps K
unrounded for CSA compression. Both short-score entry points explicitly
skip their standalone rounding when projection has already performed it.
The opt-in VLLM_STEP5_INDEXER_ROUND_IN_NORM=1 applies only to 1--128 tokens,
proxy_dim256 and BF16; longer Prefill and other dtypes keep the old path.
The formal default is still off pending a full-model screen.

The isolated reference combines the existing fused norm/RoPE and deployed
Torch casts; both timings include CUDA Graph launches. At M32/64/128,
rotary_pairs16, the complete operator improved from
0.01161/0.01334/0.01700 to 0.00412/0.00523/0.00748 ms. Rotary_pairs64
showed the same trend. Thirty-six random, zero-row and overflow-weight
cases preserved Q bits/NaN masks, K bits and signed zero. Graph replays
with changing inputs/positions also matched. Ten production tests passed,
including actual Step4Indexer projection -> verification-score integration
for queries/request1--4, partial/full rotation and long8192-region scores.
A poison cast verifies that the optimized scorer omits its redundant stage.
No full-service throughput improvement is inferred from this microbenchmark.

The formal BF16-QKV service, full target Decode Graph and ordinary MTP3,
completed the frozen 100-question zero-shot GSM8K screen at 96/100, with
no request errors or output truncations (max_tokens8192). Temperature1,
top_p0.95, seed42 and EvalScope1.12.0 native rule scoring were fixed.
This provides ordinary inference evidence for the current Graph/tile
profile; it is not a full-dataset accuracy guarantee. The calibrated-QKV
FP8 pair subsequently completed as recorded below. Evidence is preserved at
/workspace/step5-tp2-performance/fullgraph-qkv-ordinary-20261007.


### Ordinary full-Graph QKV pair and native MK overlap repair

The calibrated-QKV FP8 service completed the same frozen100 zero-shot
GSM8K screen at 97/100 versus BF16's 96/100. Both sides used ordinary
MTP3, full target Decode Graph, identical CLI, question IDs, prompt
messages, input token counts, sampling parameters and EvalScope rules.
Both completed without request errors or truncation. The new norm/round
fusion was disabled on both sides to isolate QKV precision. Custom-op
and quantization-GEMM settings were explicit true for FP8 and default true
for BF16, as checked against the installed env defaults. Worker model
inspection samples one DP's two TP workers; it is not an all-eight-worker
audit. The per-question comparison records all grade transitions.
This fixed100 result does not prove better FP8 accuracy or full-dataset
noninferiority, and the matching64K/4K FP8 service performance is unverified.
Formal QKV remains BF16 pending the performance decision.

The rejected v2 aux-stream guard also disabled the baseline MK-internal
shared/combine schedule on non-short batches. Native DeepEP-LL reports
supports_async=True, and the modular finalizer starts combine, runs shared
experts, then executes the receive hook. The revised guard limits only the
new aux stream: non-short batches retain MK-internal overlap when supported,
or remain serial when unsupported/native restrictions apply. EPLB and
two-sided-backend restrictions are preserved. A native-finalizer ordering
test confirms shared compute remains before receive completion. The repaired candidate was subsequently service-benchmarked; its result
and actual mode-selection evidence are recorded below. Twenty-eight CPU tests and four native
aux-stream/Graph GPU tests passed after the repair.

The original vllm_hcu logger inherits WARNING and drops the candidate's
INFO mode diagnostics. It now uses vllm.hcu.shared_experts, whose effective
level is INFO, and the v3 full-model screen verified both selected modes.
Evidence and test logs are preserved in the fullgraph-qkv-ordinary directory.


### Repaired auxiliary-stream result and long-context capacity candidate

The repaired v3 short-context screen (BS64, input1024/output512) completed
64/64 requests but measured 708.01256 output token/s versus 838.88914 with
the auxiliary stream off (-15.60%). Mean TPOT increased from55.18102 to
67.56472 ms (+22.44%). Input/output totals, synthetic draft/acceptance
counts and zero DP preemption deltas match the baseline. INFO logs prove
that short Decode selected the auxiliary stream and non-short batches
retained MK-internal overlap. The candidate remains disabled.
Evidence: /workspace/step5-tp2-performance/shared-overlap-v3-20261007.

In this sequence-parallel profile shared MLP gate/up/down projections set
disable_tp=True. The routed-plus-shared result uses the same final TP
all-gather in both modes; an added shared-MLP TP collective is therefore
not the cause. Auxiliary event/stream scheduling or compute contention
remain hypotheses without a full trace. The default still retains native
MK shared/combine overlap where supported.

The optional SWA Prefill crop passed ten native GPU tests, including exact
FlashAttention output equality for actual head_dim192, KVheads2 and64K
contexts. It removes only KV pages invisible to every query and reads no
device count on the CPU. It is disabled during full Graph capture.
Test evidence: /workspace/step5-tp2-performance/bs64-capacity-20261007.

The launcher accepts STEP5_GPU_MEMORY_UTILIZATION as an explicit candidate
override and rejects non-finite or out-of-range values. Defaults remain0.92
Decode and0.89 Prefill. Bash syntax and nine stub CLI cases passed. This
adds no claim of successful higher-memory startup. Sixteen requests/DP at
65536 input plus4096 output and four draft slots need1115136 whole-page
tokens/DP; the0.92 FP8 profile's994709 tokens do not meet that bound.
Higher-memory, cropped-workspace capacity must be verified by all four
initialized DP capacities, successful Graph warmup and no runtime preemption.


The0.963 capacity v1 candidate failed the startup free-memory check after
communication initialization:137.72GiB free versus138.66GiB requested.
It did not load the model or produce a performance measurement. The revised
v2 candidate uses0.955,256 scheduler tokens and capture sizes through256,
plus calibrated QKV FP8, norm/round fusion, SWA crop and explicit DSA reserve.
It passed model startup and Graph warmup. All four DP engines initialized
1205894 effective KV tokens (17.32 requests at69632), meeting1115136
page/draft tokens and the explicit1% capacity guard. Each worker reserves
1583996576 bytes for DSA side pools and fixed workspaces before KV allocation.
A single BS64 input65536/output4096 benchmark is in progress; no new TPS
result is claimed from the capacity proof.

The benchmark retains the supplied common Decode window formula:
max(first-token timestamps) to min(completed last-token timestamps).
A window observer will terminate this owned benchmark only if a measured
4096-token request finishes while another request is still queued, which
proves that common window is empty. This avoids continuing a known invalid
performance run; observation delays alone do not trigger a restart.
Candidate source and runtime evidence are under the bs64-capacity directory.


### Fused fixed-address SWA window metadata

VLLM_STEP5_FUSED_WINDOW_METADATA=1 replaces the builder's separate length,
offset, gather, mask and staging-copy operations with one Triton kernel.
It writes directly into the existing contiguous tables, one persistent table
per width, and persistent length buffers. It preserves every table element
including invisible trailing cells, accepts strided input metadata, reads
no device count on the CPU and keeps Graph replay addresses stable.
Cascade, noncausal and unsupported parallel/window cases retain native guards.
The default remains off pending a full-model performance decision.

Eight CPU policy tests passed (vendor FA dependency isolated for CPU testing),
and nineteen native GPU tests passed: queries1/4/512, window1/64/512, strided
inputs, changing sequences/pages/query offsets, persistent builder addresses
and exact production BF16 GQA Attention under Graph replay. The independent
qualification adds fifteen metadata geometries, three replays each and three
native Attention replays with long/ragged/padded requests. All outputs match
the old path exactly.
At the measured eight-request shape, old/new metadata time including staging
was0.06372/0.00819ms eager and0.03792/0.00190ms in an isolated Graph.
This removes builder overhead outside the model's FULL Graph; it is not yet
an end-to-end throughput result. Evidence: bs64-capacity-20261007.

The256-token BS64 capacity candidate did not produce a valid Decode window.
A4096-token main request completed on every DP while nine requests/DP were
still queued. Thus min(completed last token) precedes max(first token) and
the supplied benchmark's common64-request window is necessarily empty.
The owned benchmark was terminated after this proof, clients disconnected,
all running/waiting counters returned to zero, and the service was stopped.
There was no preemption. Capacity remains proven; BS64 pure-Decode TPS does
not. Preserve the original output4096 and measurement formula while improving
scheduler/workspace capacity rather than relabeling this as valid BS64 TPS.
