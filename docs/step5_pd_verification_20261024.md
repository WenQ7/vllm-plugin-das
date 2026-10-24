# Step5 source integration and Decode verification — 2026-10-24

This update retains the existing branch through `07e318f7`, including Decode
Fabric and context-bound graph fallback. It adds the measured cached ASM G1
and fixed E23 Prefill support. The tested model code is `c2daed7ef0d34ea0313eda4d9e8058a0166766cd`; the source
archive contains 925 files verified against their Git blobs. Windows archive
export explicitly disables `core.autocrlf` so Linux receives the committed bytes.

## Fresh Decode results

Four nodes x four gfx938 cards use DP4/TP4/EP16, concurrency 64 (16 per DP),
65536 input tokens and 4096 output tokens. Both `max-num-batched-tokens=32768`
and `max-model-len=1048576` are unchanged. KV is FP8 E4M3, prefix caching is off,
and the measured hot graph covers 69888 context tokens. Longer contexts retain
the synchronized eager fallback; an actual 1M-token workload was not run.

Performance uses synthetic MTP3 rates exactly
`[0.838992,0.598322,0.362686]`. One full warmup is excluded; the first three
chronologically clean formal rounds are included. Each completes 64/64 requests
with zero preemptions, inference JIT events and cache writes on all four nodes.

| Formal round | Decode tokens/s | Mean ITL (ms) | Observed acceptance |
| --- | ---: | ---: | ---: |
| 2 | 3190.035 | 56.446878 | 0.601953 |
| 3 | 3187.076 | 56.202360 | 0.597512 |
| 4 | 3156.862 | 56.675712 | 0.598287 |

Mean Decode throughput is **3177.991 tokens/s**. Mean ITL is
**56.441650 ms**, with sample standard deviation
**0.236719 ms**. All three rounds are below 60 ms.
The previous recorded runtime was
3149.197 tokens/s and 57.038124 ms; this is a fresh consolidated-source run,
not a same-session causal A/B attribution. ITL retains the original streaming
**chunk** interval definition, and must not be described as per-token TPOT.

All 16 workers verify the consolidated source and normal native extension,
and exercise Fabric dispatch/combine. The software queue cap is 4, NCCL channels
are auto, router/shared threshold 64, expert grid on, fused-router stream off,
legacy DP custom AG off. Prefill Dense/G1/G2/ASM switches are cleared for Decode.

## Separate Decode accuracy

The same code runs a separate standard-MTP3 service, with synthetic sampling
disabled. The frozen GSM8K first 100 prompts, evaluator and generation settings
match the earlier reference: concurrency 64, temperature 1.0, top-p 0.95, seed 42,
max-tokens 32768. The recorded score is **98/100**, with
**0 request errors**. The earlier reference scored 99/100.

The unchanged evaluator allows percentage-scaled numeric matches and relative
tolerance. An additional Decimal-equality check on extracted answers gives
**98/100**, without those allowances. Neither
score establishes full-dataset equivalence. Dataset SHA256 is
`c10cd5aec4ef1d542a9b4eaf0d4b7ae7a9a4b92d877e231a8a16d40a5525264c` and evaluator SHA256 is
`d449ee1f0454df42f7b65a4dcb1ad633c9bdd07af682c016355e3cf13bf8a55f`. Raw prompts and predictions remain outside Git.

## Prefill snapshot and selection

| Variant | DP/TP/EP | Prefill tokens/s | Separate standard-MTP quality |
| --- | --- | ---: | --- |
| E22 cached ASM G1 + fused G2 | 16/1/16 |42565.786| GSM8K100 96/100; long24K 15/16; long64K 16/16 |
| Fixed E23 cached ASM G1 + fused G2 | 16/1/16 |42666.451| Complete-model quality not run |

These Prefill results belong to their pinned measurement runtimes, rather than
a new full-model Prefill run of this consolidated checkout. E23 stays an explicit
performance-only candidate. Its fixed map uses the namespace after the legacy
Prefill placement; preserve the measured placement/map pair and do not apply
the legacy placement twice. The recorded pair has placement SHA256
`876ac0f5963c51d7e9e0a4629e589de70ef2d2e9ba86b9b20437d8d0beb1bb32`
and map SHA256
`6444cd5e4c7af18d51c6a660462c053248ac7274dc484403f622cf705aa61051`.

Use the existing `benchmarks/step5/launch_multinode.sh` entry point. E22 uses
`STEP5_PREFILL_PROFILE=asm_g1_1021`; E23 uses `e23_asm_1021` with explicit
16-replica and static-map settings. Existing baselines and default-off guards
remain. The reverted standalone start scripts were not restored.

SparseRS, local32/FA metadata, dynamic expert migration and the slower compact
ASM-output experiment are excluded from the delivered model code. Prefill
45000 tokens/s was not reached. See [ASM details](step5_prefill_asm_g1.md) and
[machine-readable results](../benchmarks/step5/results-pd-image1021-20261024.json).

## Validation and dependencies

Portable tests: 136 passed. The real installed Torch/vLLM lifecycle tests pass
twice per node on all four containers, and 716 Python files parse. The native
sources are unchanged from the verified R9 build, so the normal `hcu_ops`
extension is reused rather than rebuilt. Its SHA256 is
`50fbca2b1b957e5eb121d80847f044d51e3fd1c185ca4aacb7444d78db10c2d5`; there is no standalone Dense library.
Container image IDs and package versions are recorded per node in the JSON;
the fourth node has a different image ID despite matching tags and versions.

All owned verification services are stopped after collection. Local evidence
archives retain logs, source/runtime receipts and measured counters. Model
weights, native binaries, placement PT files, datasets, raw predictions and
raw traces are external assets and are not included in this commit.

The post-measurement changes are test fixtures and documentation only; serving
code is unchanged from the tested commit. AST checks now receive the ASM switch
and activation enum, subprocesses load the checkout plugin, repository paths
are independent of the working directory, and constructor tests use a local
tiny model config. The lazy-import check verifies zero new imports relative to
the target build's observed baseline rather than a hard-coded preload list.

Additional isolated checks passed: static maps 44, diagnostics 17 plus 12 subtests,
static worker bridge 10, EP tables 16, offline EPLB 37, and EP contracts 93.
Repaired complete suites: `tests/runtime_patch/test_platform_hcu_config.py` 149 passed, `tests/patch/test_worker_dispatcher.py` 21 passed.
