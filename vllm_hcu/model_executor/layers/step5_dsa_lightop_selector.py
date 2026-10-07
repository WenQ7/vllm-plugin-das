# SPDX-License-Identifier: Apache-2.0

"""Exact short-query Step5 metadata using native LightOp Top512.

Native selected IDs can be unordered. Sorting is combined with physical-region
packing and causal-tail insertion so Attention retains its original order.
"""
import os
import torch
from vllm.triton_utils import triton, tl
from vllm.logger import init_logger
from lightop import op

@triton.jit
def _prepare_scores(SCORES,NORMALIZED,P,L,
                    ROWS:tl.constexpr,QROWS:tl.constexpr,PS:tl.constexpr,
                    WIDTH:tl.constexpr,REGION:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)
    cols=tl.program_id(1)*B+tl.arange(0,B)
    score=tl.load(SCORES+row*WIDTH+cols,mask=cols<WIDTH,other=0.)
    bits=score.to(tl.int32,bitcast=True)
    # Step5's numeric sign comparison treats both zeros as a tie and treats
    # signed NaNs as nonnegative. Preserve its ordered-key behavior exactly.
    bits=tl.where(score==0.,0,tl.where(score!=score,bits&0x7fffffff,bits))
    tl.store(NORMALIZED+row*WIDTH+cols,bits.to(tl.float32,bitcast=True),mask=cols<WIDTH)
    if tl.program_id(1)==0:
        q=tl.load(P+(row%QROWS)*PS).to(tl.int32)
        visible=tl.minimum(tl.maximum(q//REGION,0),WIDTH)
        tl.store(L+row,visible)


@triton.jit
def _sort_pack(IDS,P,T,R,O,C,LO,
               QROWS:tl.constexpr,PS:tl.constexpr,TS:tl.constexpr,RS:tl.constexpr,
               OUTS:tl.constexpr,REGION:tl.constexpr,RPP:tl.constexpr,
               LOGICAL:tl.constexpr):
    row=tl.program_id(0)
    q=tl.load(P+(row%QROWS)*PS).to(tl.int32)
    req=tl.load(R+row*RS).to(tl.int32)
    current=tl.maximum(q//REGION,0)
    history=tl.minimum(current,512)
    slot=tl.arange(0,512)
    ids=tl.load(IDS+row*512+slot)
    ids=tl.sort(tl.where(ids>=0,ids,2147483647),descending=False)
    live=(slot<history)&(ids!=2147483647)
    safe=tl.where(live,ids,0)
    page=tl.load(T+req*TS+safe//RPP,mask=live,other=-1)
    physical=page*RPP+safe%RPP
    valid=tl.minimum(tl.maximum(q+1-safe*REGION,0),REGION)
    words=tl.where(live&(page>=0),physical|(valid<<24),-1)
    # The causal slot belongs only to the scalar store below. Otherwise
    # another lane could overwrite it with history padding.
    tl.store(O+row*OUTS+slot,words,mask=slot!=history)
    tl.store(O+row*OUTS+512,-1,mask=history!=512)
    if LOGICAL:
        tl.store(LO+row*OUTS+slot,tl.where(live,ids,-1),mask=slot!=history)
        tl.store(LO+row*OUTS+512,-1,mask=history!=512)
    page=tl.load(T+req*TS+current//RPP)
    physical=page*RPP+current%RPP
    valid=tl.minimum(tl.maximum(q+1-current*REGION,0),REGION)
    tl.store(O+row*OUTS+history,tl.where(page>=0,physical|(valid<<24),-1))
    if LOGICAL:
        tl.store(LO+row*OUTS+history,tl.where(page>=0,current,-1))
    tl.store(C+row,history+1)

def make_selector(device):
    # Create immutable identity-page data before capture, not per invocation.
    identity=torch.zeros((1,1),device=device,dtype=torch.int32)
    def select(logits,positions,table,requests,*,topk,region_size=8,
               regions_per_page,return_logical_ids=False,num_kv_groups=1):
        rows,width=logits.shape
        if topk!=512 or not logits.is_contiguous():
            raise ValueError("Private Top512 scope requires contiguous scores")
        if positions.numel()*num_kv_groups!=rows or requests.numel()!=rows:
            raise ValueError("Metadata shape mismatch")
        p=torch.empty((rows,513),device=logits.device,dtype=torch.int32)
        c=torch.empty(rows,device=logits.device,dtype=torch.int32)
        logical=torch.empty_like(p) if return_logical_ids else None
        if rows:
            lengths=torch.empty(rows,device=logits.device,dtype=torch.int32)
            ids=torch.empty((rows,512),device=logits.device,dtype=torch.int32)
            normalized=torch.empty_like(logits)
            _prepare_scores[(rows,triton.cdiv(width,1024))](logits,normalized,positions,lengths,
                rows,positions.numel(),positions.stride(0),width,region_size,1024,num_warps=4)
            # All score indices belong to identity page 0. Native selected
            # physical IDs are therefore logical IDs, including -1 padding.
            op.topk_transform_512(normalized,lengths,identity.expand(rows,1),ids,
                                  triton.next_power_of_2(max(1,width)),None)
            _sort_pack[(rows,)](ids,positions,table,requests,p,c,p if logical is None else logical,
                positions.numel(),positions.stride(0),table.stride(0),requests.stride(0),
                p.stride(0),region_size,regions_per_page,return_logical_ids,
                num_warps=4)
        return (p,c,logical) if return_logical_ids else (p,c)
    return select

_SELECTORS = {}
_logger = init_logger("vllm.hcu.step5.native_selector")

def supports_native_selector(logits, positions, requests, *, topk, num_kv_groups):
    return (
        topk == 512 and logits.ndim == 2
        and logits.device.type == "cuda"
        and os.environ.get("SGL_USE_LIGHTOP_TOPK_BACKAND", "0") == "0"
        and logits.dtype == torch.float32 and logits.is_contiguous()
        and 1 <= logits.shape[1] <= 16384
        and 1 <= positions.numel() <= 128
        and num_kv_groups in (1, 2, 4)
        and logits.shape[0] == positions.numel() * num_kv_groups
        and positions.ndim == 1 and positions.dtype in (torch.int32, torch.int64)
        and requests is not None and requests.ndim == 1
        and requests.dtype == torch.int32 and requests.numel() == logits.shape[0]
        and positions.device == requests.device == logits.device
    )

def prefill_sparse_meta_lightop(
    logits, q_positions, block_table, request_indices=None, *, topk,
    region_size=8, regions_per_page, return_logical_ids=False, num_kv_groups=1,
):
    if not supports_native_selector(
        logits, q_positions, request_indices, topk=topk, num_kv_groups=num_kv_groups,
    ):
        raise ValueError("Unsupported native Step5 Top512 metadata shape")
    if (block_table.dtype != torch.int32 or block_table.stride(1) != 1
            or block_table.device != logits.device):
        raise ValueError("Native Step5 selector requires int32 block-table columns")
    selector = _SELECTORS.get(logits.device)
    if selector is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Native Step5 selector requires an uncaptured warmup")
        selector = make_selector(logits.device)
        _SELECTORS[logits.device] = selector
    result = selector(
        logits, q_positions, block_table, request_indices, topk=topk,
        region_size=region_size, regions_per_page=regions_per_page,
        return_logical_ids=return_logical_ids, num_kv_groups=num_kv_groups,
    )
    _logger.info_once(
        "Step5 native LightOp Top512 causal selector active: score width=%d, "
        "positions=%s, KV groups=%d; logical IDs sorted before Attention",
        logits.shape[1], q_positions.dtype, num_kv_groups, scope="process",
    )
    return result
