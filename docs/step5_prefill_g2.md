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

Full-model synthetic performance and standard-MTP GSM validation are pending.
Keep DP16/TP1/EP16, batched-tokens8192, model-length70000, E22, native attention,
and the fixed synthetic MTP3 rates [0.838992,0.598322,0.362686] for performance.
The separate G1 path has unresolved GSM96/96 versus disabled98/98 and is not
required by G2 fusion.
