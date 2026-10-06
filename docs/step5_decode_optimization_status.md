# Step5 decode optimization status

This note records the production decisions for the single-node BW1101
`TP2 + DP4 + EP8 + MTP3` profile. Measurements use the per-channel FP8 model
and the same 64K decode topology unless a row explicitly says operator-only.

## Adopted

- DeepGEMM uses the checked-in `E=44,N=1536,device_name=BW1101.json` tuning
  file through `VLLM_TUNED_CONFIG_FOLDER`. The measured improvement is small
  but repeatable: about 0.17% on short decode and 0.24% on the matching prefill
  comparison.
- Short Step5 sparse attention uses the FlashAttention sparse path for one to
  64 query tokens. At the production geometry, single-token decode improved
  from 0.0892 ms to 0.0391 ms (2.28x), and the 16-token MTP shape improved
  from 0.1680 ms to 0.0910 ms (1.85x). A BS32 service run improved decode from
  689.32 to 696.27 token/s (about 1.0%). Long prefill remains on the tiled
  Triton path.
- The main QKV projection remains on LightOP's channel-wise FP8 hipBLASLt
  kernel. It is the fastest installed implementation for this shape.
- MTP3 short-prefill indexer scoring is batched across requests and reads the
  summary pool directly through the block table. At 64K context, the per-layer
  score stage improved from 1.2004 ms to 0.3430 ms at total BS32 (3.50x), and
  from 2.3946 ms to 0.6778 ms at total BS64 (3.53x), with bit-identical fp32
  scores.
- Q/K indexer normalization and partial RoPE share one launch when a DP has at
  most 32 target-verification tokens. The BS32 shape improved from 0.01319 ms
  to 0.00966 ms (1.37x), bit-identically. At 64 tokens the joint grid is slower,
  so BS64 and prefill retain the reference two-launch path.
- Prefill radix selection now packs physical regions and appends the current
  causal region in the same kernel. At 64K/top-512 this reduces the selection
  metadata stage from 0.1184 ms to 0.1128 ms at BS32 (5.0%), and from 0.1537 ms
  to 0.1470 ms at BS64 (4.5%), with identical packed slots, counts, and logical
  IDs.

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
- Restricting the fused main QK-norm/RoPE kernel to rotary-64 sparse layers
  was numerically close (maximum BF16 absolute difference 0.03125) and fast as
  an isolated operator. The service-level difference changed MTP behavior:
  total/output throughput fell from 1053.79/348.99 to 926.54/306.85 token/s,
  acceptance fell from 29.82% to 23.94%, and mean ITL changed from 138.68 to
  139.30 ms. The production profile therefore retains the compiled native
  QK-norm/RoPE sequence.
- Early shared-expert launch on an auxiliary stream increased the 64K warm
  prefill maximum from 153.7 s to 324.4 s and reduced the observed active
  generation rate to 22.96 token/s. The existing token threshold does not
  remove contention at the production decode shape, so the formal profile
  keeps the shared-expert stream disabled. vLLM also rejects dual-batch
  overlap when speculative decoding or CUDA graphs are enabled, so DBO cannot
  overlap DeepEP in the required MTP3 graph-enabled profile.

## Constraints that cannot be enabled by a flag

### Full target-model graph

The current fixed-address DSA schedule correctly captures one-token decode.
MTP3 target verification is a four-query prefill-shaped step. It can cross a
compression-region boundary and must preserve up to two rollback prefixes.
Changing the backend declaration from `UNIFORM_SINGLE_TOKEN_DECODE` to
`UNIFORM_BATCH` would freeze pending-row assignments and sparse score widths
from the capture step, producing wrong attention after replay. A correct full
target graph needs a persistent four-query schedule, a batched request/query
indexer scorer, and graph-stable rollback buffers. It also needs the vLLM Eagle
path to allow the draft model while the target uses a full graph. A persistent
four-query schedule and fixed-width scorer were prototyped and
passed operator-equivalence tests. Service tests still hit an HSA VM fault when
requests completed and the per-DP batch drained, both with padded graph replay
and with exact 4-token capture sizes. The target therefore stays `PIECEWISE`;
the MTP draft keeps its verified `FULL_DECODE_ONLY` graph. The short-prefill
scorer no longer specializes on the live token count and is compiled during
startup warmup, avoiding runtime Triton compilation on the first target step or
as the batch shrinks.

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

