# Optional vLLM upstream delta patches

Step5 support now lives in `vllm_hcu`. The former 24 source patches were retired;
plugin installation and startup must not rewrite the installed vLLM package.
The five files in `upstream/` preserve the vLLM commits that landed after the
old source-patch endpoint for deployments that still maintain an externally
patched vLLM tree.

The bundle is an ordered series against vLLM source commit `d0ff4a9` and
reproduces vLLM commit `12907f8` exactly:

| Patch | Source commit | Purpose |
| --- | --- | --- |
| `0001` | `684a73f` | Fuse Step5 DSA indexer q/k/z/w projection into the qkvg fp8 GEMM |
| `0002` | `7c779c1` | Use the fp8 Step5 DSA prefill indexer with causal tile skipping and resident top-k packing |
| `0003` | `6a9ad96` | Do not evict a Step4 DSA pending row retired by the previous step |
| `0004` | `f10f778` | Add static Step5 expert placement (`VLLM_STEP5_EXPERT_PLACEMENT`) |
| `0005` | `12907f8` | Accelerate the Step4 DSA prefill top-k threshold search |

The previous 0001–0024 source-patch series is historical and is no longer
shipped by this branch. If a deployment has that series applied, apply these
five patches after its `d0ff4a9` endpoint. For a clean vLLM tree, use a vLLM
checkout at `d0ff4a9` and apply only this optional delta bundle:

```bash
VLLM_DIR=/path/to/vllm-at-d0ff4a9
for patch in patches/vllm/upstream/*.patch; do
    patch -p1 -d "$VLLM_DIR" < "$patch" || break
done
```

These files are source records only. The plugin does not apply them
automatically; use the matching vLLM package and plugin revision together, and
validate the compatibility contracts after changing either one.
