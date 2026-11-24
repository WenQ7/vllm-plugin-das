# Step5 dual-node unified source

Build and install **one vllm-hcu wheel from this branch on both nodes**. The
same installation serves Native Prefill (TP1/DP16/EP16) and DeepEP/DeepGEMM
Decode (TP4/DP4/EP16). Switch profiles after stopping the current service.
No plugin source overlays, runtime loaders or edits to installed boltops are
required. Model weights, tokenizer, expert placement and tuning data remain
external deployment assets.

The Native profile keeps its FP32 Gemma norm / Torch quantization producer,
EP64 scratch and bounded DBO output pool. The Decode profile selects the
library norms, dynamic quantization and fused decode producers. The two-GiB
sorted-activation addressing fix is now part of the plugin's native EP launch;
the underlying vendor kernels remain unchanged.

## Build and launch

Use the matching Python 3.12 / Torch 2.11 / DTK 26.10 / gfx938 environment:

```bash
VLLM_HCU_BUILD_STEP5_DENSE=1 VLLM_HCU_BUILD_STEP5_PACKED=0 \
  python -m pip wheel --no-build-isolation --no-deps -w dist .
python -m pip install --no-deps --force-reinstall dist/vllm_hcu-*.whl
cp scripts/step5_dualnode/config.env.example scripts/step5_dualnode/config.env
```

Fill the two node addresses, interfaces, model/tokenizer paths, asset root and
local cache directory. Use the same wheel and asset contents on both nodes.
Prefill requires the qualified hipBLASLt build matched by its Dense recipe.
Decode requires DeepEP's layered LL/rail mapping ABI and the HCU DeepGEMM
channel-FP8 build with code-object V5 support (measured source `7da8cd9`). A
similarly named generic dependency wheel does not establish compatibility.
For containers, map `/dev/kfd`, `/dev/dri` **and `/dev/infiniband`**, use host
networking, adequate shared memory and unlimited memlock.

Required external asset layout:

```text
assets/
  expert_placement_decode.pt
  prefill/
    dense-qualified.json
    step5_ep16_moe_table.json
    expert_placement_prefill_8192_ep16_20261020.pt
    expert_placement_prefill_e23_static_tokenhash_sz_20261029.json
```

Run each command on both nodes, with `NODE_RANK=0` on the head and `1` on its
peer. Append `--dry-run` to inspect the command without loading a model:

```bash
NODE_RANK=0 bash scripts/step5_dualnode/serve.sh prefill
NODE_RANK=0 bash scripts/step5_dualnode/serve.sh decode
NODE_RANK=0 bash scripts/step5_dualnode/serve.sh decode-perf
```

`decode` uses standard MTP3 for quality testing. `decode-perf` uses synthetic
acceptance `[0.838992,0.598322,0.362686]` for the historical performance workload.
Prefill uses no MTP and explicit synchronous scheduling. No profile creates a
96-GiB physical capacity reservation. Prefill retains its historical fixed
4.75-GiB KV budget; compilation cache export is disabled to avoid filling the
system disk. These settings do not change the shared installed source.

Historical results from separate releases are provenance, not validation of
this unified wheel. Record the wheel SHA256 and commit for both profiles,
then check performance and fixed100 GSM8K independently with the corresponding
speculation settings.

The launcher enables the read-only `step5_source_identity` worker RPC. Query
`/collective_rpc` with `{"method":"step5_source_identity","timeout":30}` and
compare installed versions, native-extension hashes and critical source hashes
on all workers for both profiles. This detects stale wheels or source overlays.
