"""Quantize occupied Step5 expert rows without a per-expert launch grid."""

from vllm.triton_utils import tl, triton


@triton.jit
def compact_activated_quant(q, scale, activation, counts, capacity: tl.constexpr,
                            width: tl.constexpr, experts: tl.constexpr, chunk: tl.constexpr,
                            min_flat_row: tl.constexpr = False, fused_row_max: tl.constexpr = False):
    expert_ids = tl.arange(0, 64)
    rows = tl.load(counts + expert_ids, mask=expert_ids < experts, other=0)
    rows = tl.maximum(tl.minimum(rows, capacity), 0)
    inclusive = tl.cumsum(rows, axis=0)
    total = tl.sum(rows, axis=0)
    cols = tl.arange(0, chunk)
    for active_row in range(tl.program_id(0), total, tl.num_programs(0)):
        if min_flat_row:
            tl.static_assert(64 * capacity < 0x7fffffff)
            # Consecutive candidates differ by capacity minus the preceding
            # expert's row count. The first eligible lane is therefore minimal.
            candidate = expert_ids * capacity - (inclusive - rows) + active_row
            flat = tl.min(tl.where(active_row < inclusive, candidate, 0x7fffffff), axis=0).to(tl.int64)
        else:
            expert = tl.sum((inclusive <= active_row).to(tl.int32), axis=0)
            preceding = tl.sum(tl.where(expert_ids == expert, inclusive - rows, 0), axis=0)
            flat = expert.to(tl.int64) * capacity + active_row - preceding
        offset = flat * width + cols
        y0 = tl.load(activation + offset)
        y1 = tl.load(activation + offset + chunk)
        y2 = tl.load(activation + offset + 2 * chunk)
        if fused_row_max:
            peak = tl.maximum(tl.maximum(tl.abs(y0), tl.abs(y1)), tl.abs(y2))
            amax = tl.max(peak, 0)
        else:
            amax = tl.maximum(tl.maximum(tl.max(tl.abs(y0), 0), tl.max(tl.abs(y1), 0)), tl.max(tl.abs(y2), 0))
        scale_value = tl.maximum(amax.to(tl.float32), 1.0e-10) / 448.0
        inverse = 1.0 / scale_value
        tl.store(q + offset, (y0 * inverse).to(q.dtype.element_ty))
        tl.store(q + offset + chunk, (y1 * inverse).to(q.dtype.element_ty))
        tl.store(q + offset + 2 * chunk, (y2 * inverse).to(q.dtype.element_ty))
        tl.store(scale + flat, scale_value)
