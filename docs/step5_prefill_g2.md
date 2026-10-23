# Step5 prefill G2 fusion

`HCU_STEP5_PREFILL_G2=1` enables a normal Python/Triton source path for the
fixed gfx938 E22, M65536, topk8, sort-BM256 channel-FP8 case. Default is off.
The normal `hcu_ops` library and the model weights do not change.

The existing Directsum metadata and scratch-overflow decision feed one G2
launch. It stores either to the Directsum destination or the original route
cache. The native final fix kernel retains the accumulation order. Unsupported
metadata/configurations retain the native two-launch path. Input/output offsets
use 64 bits, including the >2 GiB full-route overflow case.

Three isolated route levels, eight independent weight/input pairs, four
amplitudes, empty routes and forced overflow produced 102 complete-MoE bitexact
comparisons. Paired complete-MoE latency ratios were 0.94764/0.95510/0.96122
for p10/p50/p90; graph ratios were 0.94843/0.95519/0.96113.
These are operator measurements, not full-model throughput.

Run `benchmarks/step5/check_prefill_g2_fused.py --run-gpu --fixture FIXTURE.npz
--fixture-meta FIXTURE.json --output RESULT.json` in the normal package environment
to compare the integrated source, including device overflow and unsupported
metadata guards. Use the existing frozen routing fixture format.

With G1 specialization disabled, three clean full-model synthetic rounds gave
42,224.93 tokens/s (sample SD 8.25), 1.55% above the earlier 41,580.14 result.
DP16/TP1/EP16, batched-tokens8192, model-length70000, E22, native attention,
C64 x 65536 input tokens x 1 output token, and synthetic MTP3 rates
[0.838992,0.598322,0.362686] were retained. There were no preemptions or
in-measurement compilation/cache updates. This remains below 45,000 tokens/s.

Separate standard-MTP evaluation on the same delivered source scored GSM100
97/100, long24K16 15/16, and long64K16 15/16, with zero request errors. The
short GSM prompts total only 8,485 tokens (maximum 154 each); they cannot
exercise the M65536 specialization, so this sample alone does not establish
its quality effect. Long-context scores match the earlier 15/16 results;
evaluation-window dispatch counts were not collected. The independent
integrated complete-MoE check passed 34 bitexact comparisons and nine metadata
fallback guards. All performance and quality services were stopped afterward.

The separate G1 path is not required by G2 fusion. Prior stochastic short-GSM
scores of G1-enabled96/96 versus disabled98/98 need coverage-aware evaluation
before attributing a quality difference to that specialization.
