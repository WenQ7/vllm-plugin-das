# SPDX-License-Identifier: Apache-2.0
"""Host-only phase guard for Step5 shared-expert overlap."""

def is_short_uniform_decode(context):
    """Use the scheduler's host descriptor; never inspect a device tensor."""
    if context is None:
        return False
    cached = getattr(context, "_step5_short_uniform_decode", None)
    if cached is not None:
        return cached
    desc = getattr(context, "batch_descriptor", None)
    ready = False
    if desc is not None and desc.uniform:
        tokens = desc.num_tokens
        requests = desc.num_reqs
        ready = bool(requests is not None and requests > 0 and tokens > 0
                     and tokens % requests == 0 and 1 <= tokens // requests <= 4)
    context._step5_short_uniform_decode = ready
    return ready
