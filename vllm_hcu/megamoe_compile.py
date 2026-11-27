# SPDX-License-Identifier: Apache-2.0
"""Keep rank-coordinated MoE calls out of independent Inductor autotuning."""
import threading
import weakref

_BOUND={}
_LOCK=threading.RLock()

MEGAMOE_SPLITTING_OPS=(
    'vllm::moe_forward',
    'vllm::moe_forward_shared',
    'vllm::step5_moe_forward_output',
)


def configure_megamoe_splitting(config):
    kernel=getattr(config,'kernel_config',None)
    backend=getattr(kernel,'moe_backend',None)
    backend=getattr(backend,'value',backend)
    if backend!='deep_gemm_mega_moe':return False
    if getattr(getattr(config,'model_config',None),'enforce_eager',False):return False
    compilation=config.compilation_config
    key=id(compilation)
    def remove(reference):
        with _LOCK:
            if _BOUND.get(key) is reference:_BOUND.pop(key,None)
    with _LOCK:_BOUND[key]=weakref.ref(compilation,remove)
    # The platform hook can precede the default splitting-op finalizer.
    # Bind without replacing None; the official callback applies the policy.
    return apply_bound_megamoe_splitting(compilation)


def apply_bound_megamoe_splitting(compilation):
    with _LOCK:
        reference=_BOUND.get(id(compilation))
        if reference is None or reference() is not compilation:return False
    ops=compilation.splitting_ops
    if ops is None:return False
    if not isinstance(ops,list):
        raise ValueError('Finalize the default splitting ops before adding MegaMoE collectives')
    changed=False
    for name in MEGAMOE_SPLITTING_OPS:
        if name not in ops:ops.append(name);changed=True
    return changed
