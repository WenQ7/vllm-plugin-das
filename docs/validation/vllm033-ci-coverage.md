# v0.33 CI coverage candidate

A change to `adapter_extension/src/hcu_vllm_das/environment.py` or
`attention/e5m2_scales.py` previously selected no HCU jobs and did not trigger
fallback. The legacy workflows only target `v0.28.1-dev`.

`hcu-pr-ci-v033.yml` handles PRs and pushes to `v0.33.0` and dispatches on that
release. It invokes the existing selector CLI with `--release v0.33.0` and the
independent `hcu-test-map-v033.json`. The default CLI, legacy map, registry and
workflows retain the v0.28.1 behavior. It never sends v0.33 changes to legacy
vllm_hcu jobs or substitutes the v0.28.1 image.

Every mapped code change selects the v0.33 source and device contracts.
Every unmapped non-documentation path selects the same scope and is reported in
`unmapped_paths` with `fallback=true`. Deleted paths are included in release
diff selection. Documentation alone selects no jobs; `--full` and
`--accuracy` expand to the full declared v0.33 scope even for empty/doc diffs.
The CI selector/map/workflow and test tooling also select the full scope.
An explicit `--config` is passed to the selected release's parser; the default
map is chosen by release only when no override is provided. Invalid overrides
fail selection rather than silently using the built-in map.

PR hardware execution reuses the existing author/`ready-hcu` authorization
helper from the PR's trusted base commit on a hosted CPU runner. A trusted
author is authorized automatically; other authors need `ready-hcu` applied
by a trusted actor. Label additions/removals trigger a new check. CPU source
contracts run independently. Release pushes/manual runs must target
`refs/heads/v0.33.0`. Image resolution and the self-hosted device job depend
on authorization. The aggregate gate rejects a selected but unauthorized,
skipped, failed or cancelled stage, so a skipped hardware job is not a pass.

The source job runs existing official API AST and package-boundary tests using
the immutable official target `4ba787ec9959e566d3b587a796dff07673324da5`.
It does not certify the whole portable suite, native ABI, or model quality.
Portable collects ordinary contract files and uses `-m 'not hcu'` with devices
hidden. Device uses AST marker discovery without importing pure portable modules,
then `-m hcu` with the live flags. Module/class marks and decorated cases also
cover a file containing
portable format contracts and native numerical cases together, including
SlimQuant's six marked cases in `test_slimquant.py`. Device enables the existing
`HCU_SLIMQUANT_LIVE=1` opt-in. Deselected cases belong to the other tier; executed
cases must still contain no skips or xfails. Script-only selector cases retain
their eleven explicit subprocess invocations outside pytest. The job
installs only the independent `adapter_extension` project, never root
`setup.py`, and explicitly selects the Adapter platform entry point in every
tier so an image's legacy platform entry point cannot be discovered. The
installed tiers pin Core, Adapter and Plugin packages before test imports and
verify loaded module origins against their snapshotted non-editable RECORDs
after collection and at session end. A pure portable module's source-path
fixture is never imported just to select device cases. These are file-ownership
checks, not a replacement for wheel integrity, provider or numerical validation.
The
workflow isolates parent Python discovery and clears inherited `PYTHONPATH`
and user site packages from child environments. Empty
results, failures, errors, skips/xfails and unexpected passes
fail the result gate. JUnit reports preserve each tier separately. The shared
HCU registration assertion recognizes only files actually executed by this
release runner; all other HCU files still require the legacy registry.

Installed-version and source-root discovery use a separate process with the
same interpreter, sanitized environment and working directory as the tests.
The probe queries metadata and the module spec without importing vLLM; both
the release gate and portable AST root use that child's result. A controller's
different `sys.path` or inherited `PYTHONPATH` cannot choose the tested Core.

Set `HARBOR_SITE` to the approved registry host (optionally with a port, without
a scheme or path). Set `HCU_CI_IMAGE_V033` to an immutable
`${HARBOR_SITE}/jenkins/model_test_env/vllm@sha256:<64 lowercase hex digits>`
for a prepared DTK image with compatible vLLM 0.33, Adapter, vendor libraries,
pytest, packaging and build tooling. Missing/whitespace values, mutable tags,
malformed digests and other repositories fail before the hardware job starts;
there is no legacy-image fallback or automatic pull to resolve a tag.
The job container uses the existing `HARBOR_USERNAME`/`HARBOR_PASSWORD` secrets
when the event permits them, for the initial private-image pull. Fork PRs
normally cannot read repository secrets: a private-image pull therefore still
requires reviewed pre-existing runner registry authentication or credentials
available to that event. The credentials declaration does not prove pull success.
Missing authorization or pull access remains a failure; do not switch to
`pull_request_target`, expose secrets to PR code, or let a PR head authorize itself.
### ARC rollout for BW1100 / gfx938

Set the repository Actions variable `HCU_CI_V033_K8S_MODE` to choose the runner
for gfx938:

| Mode | gfx938 execution | Required by `v033-ci-gate` |
| --- | --- | --- |
| Unset or `legacy` | Existing BW1100 runner | Legacy gfx938 job |
| `compare` | Legacy job plus ARC one- and two-card jobs | Legacy gfx938 job; review the ARC job results separately |
| `k8s` | ARC one- and two-card jobs | Both ARC matrix entries must pass |

Start with `compare` and compare the same DeepGEMM numerical contract on the same
image digest before switching to `k8s`. The ARC jobs run inside the resolved
job container: no nested Docker, manual device mappings, mounts, or visibility
mask. The runner verifies the visible card count and architecture. The existing
BW1000/gfx936 job (including its portable tier), authorization and CPU source
gate remain on their existing runners. The legacy device containers retain the
read-only `/opt/hyhal` mount and 32 GiB shared memory.

| ARC pool | Workload | Job timeout |
| --- | --- | --- |
| `hygon-bw1100-1card` | Existing Channel-FP8 DeepGEMM numerics | 60 min |
| `hygon-bw1100-2card` | Single-node RCCL all-gather and all-reduce on two ranks | 90 min |

The two-card device contract is a communication smoke test, separate from
real model checks. Job timeouts do not give GitHub's queued state a deadline;
ARC also has its own Pod startup limits.

JUnit reports from the ARC entries have separate artifact names and 30-day
retention. The hosted selector/source jobs receive no Harbor secrets or
HCU devices. PR-code jobs use read-only contents permissions and disable
persisted git credentials, including the official source checkout.

### Real-model checks in v0.33 PR CI

For a selected executable change, the authorized `v033-models-k8s` matrix runs
in the same immutable v0.33 image, regardless of the device comparison mode.
The existing gate requires the model matrix to pass. This uses the v0.33
Core/Adapter/Plugin path and never installs the v0.28.1 project.

| ARC pool | Checks | Job timeout |
| --- | --- | --- |
| `hygon-bw1100-1card` | Qwen3.5-9B and Qwen3-8B GSM8K; Qwen3-4B LoRA A → B → A | GSM8K 90 min; LoRA 60 min |
| `hygon-bw1100-2card` | Qwen3.5-35B-A3B TP2/EP2 | 90 min |
| `hygon-bw1100-4card` | Qwen3.5-35B-A3B TP4/EP4 | 90 min |
| `hygon-bw1100-8card` | DeepSeek-R1 Channel-INT8 TP8/EP8; DeepSeek-R1 Channel-FP8 GSM8K | TP/EP 240 min; GSM8K 360 min |

GSM8K uses the fixed first 100 test questions and the 95% pass threshold,
including exact prediction, review and report counts. EvalScope 1.9.1 is
installed in the disposable evaluation job only, with the Core and Adapter
packages protected. TP/EP checks the live engine topology and token generation
using v0.33's backend selection. It does not force v0.28.1-specific kernels.

The pools supply read-only model storage. By default model paths are looked up
under `/public/opendas/DL_DATA/llm-models` and
`/public4/opendas/DL_DATA/llm-models`; GSM8K expects local train/test parquet
under the corresponding `ci_datasets/gsm8k` path. Configure
`HCU_CI_V033_MODEL_ROOTS` (colon-separated) or `HCU_CI_V033_GSM8K_PATH` only
when the pool uses a different layout. Missing resources fail the check;
the workflow does not download models or data. The model reports, logs and
JUnit files are uploaded as GitHub artifacts for 30 days.

A configured digest is not evidence that the image or hardware tier passed.
Each device pool must provide its documented runtime, the requested devices,
and access to the prepared DTK image. This workflow does not create an image
or certify models and other HCU architectures. Image/wheel provenance, model quality, performance
and the remaining release matrix remain separate acceptance gates.

Installed tiers use the existing `packaging` dependency to parse PEP 440 epoch
and release tuples, preserving the runtime activation policy for rc/dev/post/local
builds and explicit epoch zero. The import is lazy so source/selection bootstrap
still works on a stdlib-only interpreter. Script-only public-selector contracts
run as eleven explicit isolated cases; they are not counted as collected pytest
tests. Installed tiers default to `TORCHINDUCTOR_COMPILE_THREADS=1`, while keeping
explicit overrides, torch.compile and changed-input Graph contracts enabled.

CPU verification commands:

```bash
python -I -B -m unittest discover -s tests/ci -v
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HCU_ADAPTER_HOST=none \
  HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
  python -I -B -m pytest -q tests/ci tests/patch/test_hcu_ci_selector.py
VLLM_TARGET_ROOT=/path/to/official-v033 \
  python -I -B tools/run_v033_ci.py --tier source --junit v033-source.xml
```

The upgrade audit preserves the original zero-job probe and the first full
regression failure: the old global registration assertion did not recognize
three existing v0.33 HCU-marked files. The candidate accounts for them via the
actual release runner rather than weakening registration for other files.
Local CPU results do not certify the ARC pool, model or hardware tests.

The integration follow-up incorporates the merged optional SlimQuant format
from MR363. Its generic public Core PLE interface remains an enablement and
deployment dependency; the CI wrapper does not invent that API, override its
capability guard, or count a missing interface/skip as positive device evidence.
