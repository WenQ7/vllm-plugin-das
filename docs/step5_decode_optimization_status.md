# Step5 decode optimization status

This note records the production decisions for the single-node BW1101
`TP2 + DP4 + EP8 + MTP3` profile. Measurements use the per-channel FP8 model
and the same 64K decode topology unless a row explicitly says operator-only.

## Adopted

- DeepGEMM uses the checked-in `E=44,N=1536,device_name=BW1101.json` tuning
  file through `VLLM_TUNED_CONFIG_FOLDER`. The measured improvement is small
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
- The main QKV projection remains on LightOP's channel-wise FP8 hipBLASLt
  kernel. It is the fastest installed implementation for this shape.
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

### Full target-model graph

The target model can now capture all six uniform MTP3 graph sizes. Its
four-query DSA schedule keeps fixed addresses for compression, rollback
prefixes, request indices, and score storage, and is bit-identical to the
general scheduler. A 32-request drain test with staggered 16--78 token outputs
completed 32/32 requests without a VM fault or worker failure.

The one-run service A/B did not justify enabling it: the same synthetic workload
took 9.56 seconds with `FULL_AND_PIECEWISE` and 6.67 seconds with `PIECEWISE`, a
43.3% regression. The completion-drain interval was essentially unchanged
(2.92 versus 2.94 seconds), so the full target graph did not improve steady
decode and added initial latency. The formal launcher therefore retains target
`PIECEWISE`; the MTP draft keeps its verified full decode graph.

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
`N=6912,K=4096` entry. Its public channel-wise hipBLASLt path is already the
production backend and remains faster than the installed AITER alternatives in
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

