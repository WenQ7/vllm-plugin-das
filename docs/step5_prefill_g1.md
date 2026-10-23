# Step5 prefill G1 specialization

`HCU_STEP5_PREFILL_G1=1` enables a shape-specialized Triton projection in the
existing channel-FP8 EP path. It consumes the original contiguous NK weights
and expert-sorted buffers. FP32 scaling, the BF16 projection boundary,
SwigluStep, activation quantization, GEMM2 and Directsum remain unchanged.
There is no additional native library or worker extension.

The implementation is limited to gfx938, E22, N3072/K4096, top-k 8, a 256-row
sorting block and a global MoE microbatch of 65536 tokens. The latter is the
DP16 DBO case; the per-DP `--max-num-batched-tokens` remains 8192. Unsupported
shapes, layouts, dtypes and devices use the existing projection. The feature
is disabled by default and allocates no temporary tensors or weight copy.

The source benchmark runs eight independent FP8 weight/input pairs through
the complete native MoE pipeline. It checks changed inputs, zero inputs,
activation saturation, an empty local route set and Directsum overflow before
paired eager and CUDA graph timing:

```bash
python benchmarks/step5/check_prefill_g1_specialized.py \
  --run-gpu --fixture fixture.npz --fixture-meta fixture.json \
  --output /tmp/step5-prefill-g1-result.json
```

The fixture contains `[65536, 8]` physical expert IDs; its metadata identifies
the local E22 range and verifies the NPZ SHA256. Run this isolated benchmark
on a spare gfx938 GPU. It reports operator latency, not model tokens/s.

On the 1021 image, three frozen routing distributions passed 102 bit-exact
complete-MoE comparisons with approximately 2.5–2.7% lower MoE latency. The
normal-package entry passed another 34 bit-exact comparisons and measured
6.953 ms versus 6.775 ms on
the middle routing distribution. Separate projection tests cover all-local
routing and sorted addresses beyond 2 GiB. Four-node model performance and
standard-MTP GSM8K validation remain pending; these operator results do not
establish the 45,000 tokens/s target.
