# Step5 W8A8 MegaMoE source integration

This branch builds on HYGON-AI/vllm-plugin-das `v0.28.1-dev` at
`52c7bf49418ca59089aa7bcbd9ef9931ca626edb`. DeepGEMM builds on the internal
`develop` branch at `795b1b20d61b1716eb9cc2e3efd04c03a57d9b41`.

## Build

Build and install the accompanying DeepGEMM source branch first, using HIP
PyTorch and DTK on Linux:

```bash
PYTORCH_ROCM_ARCH=gfx938 MAX_JOBS=2 pip wheel . --no-build-isolation --no-deps -w dist
pip install --force-reinstall --no-deps dist/deepgemm-*.whl
```

The normal DeepGEMM wheel contains `deepgemm.op`, stock `deepgemm.mega._C`,
and `deepgemm.mega_step5._C_step5`. The last is an ordinary source-built
package extension, with a separate translation unit and isolated namespace.
There is no manual `.so` copy or external-library-path setting. The regular
vllm-hcu build compiles its existing `hcu_ops` extension as before.

From this plugin checkout, use the repository's normal build/install procedure.
Do not add the previous experiment's `plugin-allrows`, `deepgemm-python`,
probe directories or worker-extension modules to `PYTHONPATH`.

## Runtime opt-in

In addition to the existing Step5 Channel-FP8/DeepEP deployment configuration:

```bash
export HCU_STEP5_PAIR32_MEGAMOE_DECODE=1
export HCU_STEP5_MEGAMOE_ALL_DECODE=1
export DG_MEGA_STEP5_PHASED_VARIANT=g2-fused-completion
export DG_MEGA_PEER_MEMORY=ipc
```

This path requires Step5, gfx938, single-node EP8/TP4 with direct peer memory,
352 experts, top-k8, hidden4096 and intermediate1536. Supported uniform
DP row counts are16/32/64/128, corresponding to4/8/16/32 local source tokens.
Without `HCU_STEP5_MEGAMOE_ALL_DECODE=1`, the existing64-row integration is
retained. Without the MegaMoE flag, other models and the ordinary backend
selection stay on the upstream path. The custom-op master switch also gates
the feature.

Forward context carries the already-coordinated DP counts. All ranks reject
uneven or unsupported counts together and use ordinary DeepEP/DeepGEMM.
This integration does not need the earlier peer-write DeepEP fork, native-G2
publication operator, probe `.so` loaders or experimental source overlays.
DeepEP remains a normal dependency for the fallback path.

The64-row native TP4 fused output is retained. Other supported row counts
use the existing IPC all-gather and normal shared-expert reduction/scaling.
Shared experts and model weights are not incorporated into the MegaMoE kernel.

## Weight layout

The new DeepGEMM upstream default is Pair32 v2. Its tensor shape is identical
to original Pair32, but its bytes differ. Step5 explicitly selects
`pack_fp8_weight_pair32`, records the layout on the layer, and passes
`weight_layout="pair32"` to G1/G2 fallback GEMMs. Other callers retain the
upstream v2 default. A missing layout marker fails initialization rather than
running the Step5 kernel on ambiguously packed weights.

## Validation

```bash
pytest tests/runtime_patch/test_step5_megamoe_contract.py
pytest tests/runtime_patch/test_moe_deepep.py
```

DeepGEMM supplies CPU layout/build contracts, stock-source isolation checks,
Pair32 epilogue tests and eight-rank replay tests under `op_test/step5_megamoe`.
Validation of this rebased source build must be reported separately from the
previous experiment's prebuilt binary performance numbers.
