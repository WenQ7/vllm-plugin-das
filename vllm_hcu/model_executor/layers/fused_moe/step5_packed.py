# SPDX-License-Identifier: Apache-2.0
"""Pair32 FP8 experts using the package's native HCU operators.

Weights retain their original parameter shape and storage size. The target
model is packed once after loading; draft weights keep their original layout.
"""
import torch
import vllm_hcu.hcu_ops  # Registers the compiled operators.
from vllm.triton_utils import triton, tl
from vllm_hcu.model_executor.layers.fused_moe import ep_fused_moe as native
from vllm_hcu.model_executor.layers.fused_moe.ep_fused_moe import _ep_swiglu_chunk
from vllm.logger import init_logger

FP8=torch.float8_e4m3fn
PACKED={}
logger=init_logger("vllm.hcu.step5")

def _fake_packed_gemm(a,weight,a_scale,weight_scale,indices,output):
    return None
for _name in ["step5_fp8_packed_masked","step5_fp8_packed_contiguous"]:
    torch.library.register_fake("hcu_ops::"+_name)(_fake_packed_gemm)

def weight_key(w1,w2):
    return (w1.device.index,w1.data_ptr(),w2.data_ptr())

def is_packed(w1,w2):
    return weight_key(w1,w2) in PACKED

def _packed_view(weight):
    e,n,k=weight.shape
    return weight.view(e,k//64,n//32,2,4,16,16)

def _pack_weight(weight):
    e,n,k=weight.shape
    sn,sk=weight.stride()[1:]
    packed=torch.empty((e,k//64,n//32,2,4,16,16),device=weight.device,dtype=weight.dtype)
    for expert in range(e):
        source=torch.as_strided(weight[expert],size=(n//32,2,4,4,k//64,4,16),
            stride=(32*sn,sn,8*sn,2*sn,64*sk,16*sk,sk))
        packed[expert].copy_(source.permute(4,0,1,5,3,2,6).contiguous().view_as(packed[expert]))
    return packed

@torch.inference_mode()
def prepare_model_weights(worker, *, reloaded=False):
    config=worker.vllm_config;p=config.parallel_config
    from vllm_hcu.runtime_compat.step5 import is_step5_config
    if not (is_step5_config(config) and p.tensor_parallel_size==p.data_parallel_size==4
            and p.enable_expert_parallel and p.all2all_backend=="allgather_reducescatter"
            and not p.enable_dbo and native.CHUNK==16384
            and worker.model_runner.dtype==torch.bfloat16):
        return
    if not torch.cuda.get_device_properties(worker.device).gcnArchName.startswith("gfx938"):
        return
    pairs=[];seen=set()
    for module in worker.model_runner.model.modules():
        w1=getattr(module,"w13_weight",None);w2=getattr(module,"w2_weight",None)
        if not isinstance(w1,torch.Tensor) or not isinstance(w2,torch.Tensor):continue
        if w1.shape!=(22,3072,4096) or w2.shape!=(22,4096,1536):continue
        if w1.dtype!=FP8 or w2.dtype!=FP8:continue
        key=weight_key(w1,w2)
        if key not in seen:pairs.append((w1,w2));seen.add(key)
    if not pairs:return
    if len(pairs)!=88:raise RuntimeError("Unsupported Step5 packed expert layout")
    before=torch.cuda.memory_allocated(worker.device)
    for w1,w2 in pairs:
        key=weight_key(w1,w2)
        if key in PACKED and not reloaded:continue
        for weight in [w1,w2]:
            if not weight.is_contiguous():raise RuntimeError("Step5 FP8 weights must be contiguous")
            packed=_pack_weight(weight)
            weight.copy_(packed.view_as(weight))
            del packed
        PACKED[key]=(_packed_view(w1),_packed_view(w2))
    torch.cuda.synchronize(worker.device)
    if torch.cuda.memory_allocated(worker.device)!=before:
        raise RuntimeError("Step5 packing retained additional device storage")
    torch.cuda.empty_cache()
    native._step5_packed_dispatch = True
    logger.info("Prepared %d Step5 pair32 expert layers without additional weight storage",len(pairs))

def _masked_gemm(a,b,output,counts,expected):
    torch.ops.hcu_ops.step5_fp8_packed_masked(a[0],b[0],a[1].contiguous(),b[1].contiguous(),counts,output)

def _contiguous_gemm(a,b,output,indices):
    torch.ops.hcu_ops.step5_fp8_packed_contiguous(a[0],b[0],a[1].contiguous(),b[1].contiguous(),indices.contiguous(),output)

@triton.jit
def prepare_masked(A, AS, SIDS, LAYOUT, MA, MAS, COUNTS, INDEX, M:tl.constexpr):
    e=tl.program_id(0)
    lane=tl.program_id(1)
    start=tl.load(LAYOUT+e*2)*16
    padded=tl.load(LAYOUT+e*2+1)*16
    col=tl.arange(0,4096)
    if lane==0:
        row=tl.arange(0,256)
        sid=tl.load(SIDS+start+row,row<padded,other=M|(8<<24))
        tl.store(COUNTS+e,tl.sum(((sid&0xFFFFFF)<M).to(tl.int32),0))
    for row in range(lane,padded,32):
        sid=tl.load(SIDS+start+row)
        token=sid&0xFFFFFF
        if token<M:
            value=tl.load(A+token*4096+col)
            tl.store(MA+(e*256+row)*4096+col,value)
            tl.store(MAS+e*256+row,tl.load(AS+token))
            tl.store(INDEX+token*8+(sid>>24),e*256+row)


@triton.jit
def masked_quant(Q, QS, C1, COUNTS, SW, LAYOUT):
    e=tl.program_id(0)
    lane=tl.program_id(1)
    count=tl.load(COUNTS+e)
    start=tl.load(LAYOUT+e*2)*16
    col=tl.arange(0,512)
    for row in range(lane,count,32):
        flat=e*256+row
        offs=flat.to(tl.int64)*3072+col
        y0=_ep_swiglu_chunk(C1,offs,True,7.,1536)
        y1=_ep_swiglu_chunk(C1,offs+512,True,7.,1536)
        y2=_ep_swiglu_chunk(C1,offs+1024,True,7.,1536)
        amax=tl.maximum(tl.maximum(tl.max(tl.abs(y0),0),tl.max(tl.abs(y1),0)),tl.max(tl.abs(y2),0))
        scale=tl.maximum(amax.to(tl.float32),1.e-10)/448.
        inv=1./scale
        w=tl.load(SW+start+row)
        qoff=flat.to(tl.int64)*1536+col
        tl.store(Q+qoff,(y0*inv).to(Q.dtype.element_ty))
        tl.store(Q+qoff+512,(y1*inv).to(Q.dtype.element_ty))
        tl.store(Q+qoff+1024,(y2*inv).to(Q.dtype.element_ty))
        tl.store(QS+flat,scale*w)


@triton.jit
def contiguous_metadata(SIDS,NVALID,IDS,EMAP,MIDX,INDEX,
                        M:tl.constexpr,ROWS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    n=tl.load(NVALID)
    active=(row<ROWS)&(row<n)
    # Each nonempty BM256 block begins with a real route to its expert.
    # Derive the local ID through the actual map, including noncontiguous maps.
    first=(row//256)*256
    sid0=tl.load(SIDS+first,active,other=0)
    token0=sid0&0xFFFFFF
    slot0=sid0>>24
    global_e=tl.load(IDS+token0*8+slot0,active,other=0)
    local_e=tl.load(EMAP+global_e,active,other=-1)
    tl.store(MIDX+row,tl.where(active,local_e,-1),row<ROWS)
    sid=tl.load(SIDS+row,active,other=M|(8<<24))
    token=sid&0xFFFFFF
    slot=sid>>24
    tl.store(INDEX+token*8+slot,row,active&(token<M)&(slot<8))


@triton.jit
def indexed_sum(C2,INDEX,OUT,IDS,EMAP,ADD,STRIDE_ADD,ADD_START,ADD_END,
                BLOCK_N:tl.constexpr,SCALE:tl.constexpr,HAS_ADD:tl.constexpr):
    token=tl.program_id(0)
    col=tl.program_id(1)*BLOCK_N+tl.arange(0,BLOCK_N)
    acc=tl.zeros((BLOCK_N,),tl.float32)
    for slot in tl.static_range(8):
        expert=tl.load(IDS+token*8+slot)
        if tl.load(EMAP+expert)>=0:
            row=tl.load(INDEX+token*8+slot)
            acc+=tl.load(C2+row.to(tl.int64)*4096+col).to(tl.float32)
    if SCALE!=1.:
        acc*=SCALE
    if HAS_ADD:
        if (token>=ADD_START)&(token<ADD_END):
            acc+=tl.load(ADD+(token-ADD_START).to(tl.int64)*STRIDE_ADD+col).to(tl.float32)
    tl.store(OUT+token.to(tl.int64)*4096+col,acc.to(OUT.dtype.element_ty))


def shared_scratch(workspace,rows,out,avoid):
    # One pool serves C1 and then C2; their lifetimes do not overlap.
    pool=native._carve(workspace,(rows,4096),torch.bfloat16,out.device,avoid=avoid)
    return pool.view(-1)[:rows*3072].view(rows,3072),pool


def quant_scratch(workspace,rows,device,avoid=None):
    # Gathered G1 input is dead before Q2 overwrites the same pool.
    buf=native._carve(workspace,(rows*4096+4*rows,),torch.uint8,device,avoid=avoid)
    a=buf[:rows*4096].view(FP8).view(rows,4096)
    asc=buf[rows*4096:rows*4096+rows*4].view(torch.float32).view(rows,1)
    q=buf[:rows*1536].view(FP8).view(rows,1536)
    qs=buf[rows*1536:rows*1536+rows*4].view(torch.float32).view(rows,1)
    return a,asc,q,qs


def apply_chunk(a,w1,w2,weights,ids,emap,s1,s2,*,a1_scale=None,out=None,
                output_dtype=None,gemm1_limit=None,routed_scaling_factor=1.,
                configs=None,expert_mask=None,workspace13=None,workspace2=None,
                preserve_chunk_output=None):
    m=a.shape[0]
    assert a.shape==(m,4096) and ids.shape==weights.shape==(m,8)
    assert configs is None and float(gemm1_limit or 7.)==7.
    assert (output_dtype or (out.dtype if out is not None else torch.bfloat16))==torch.bfloat16
    assert emap.shape==(352,) and ids.is_contiguous() and weights.is_contiguous()
    assert s1.shape in ((22,3072),(22,3072,1)) and s2.shape in ((22,4096),(22,4096,1))
    assert s1.dtype==s2.dtype==torch.float32
    if out is None:
        out=torch.empty((m,4096),dtype=torch.bfloat16,device=a.device)
    assert out.shape==(m,4096) and out.dtype==torch.bfloat16 and out.is_contiguous()
    if m==0:
        return out
    if a.dtype in (torch.bfloat16,torch.float16):
        a,a1_scale=native.per_token_quant_triton(a,quant_dtype=FP8)
    assert a.dtype==FP8 and a1_scale is not None and a1_scale.numel()==m
    a1_scale=a1_scale.view(m,1)
    p1,p2=PACKED[weight_key(w1,w2)]
    avoid=preserve_chunk_output if preserve_chunk_output is not None else out
    index=torch.empty((m,8),dtype=torch.int32,device=a.device)
    if m<=256:
        layout=torch.empty((22,2),dtype=torch.int32,device=a.device)
        mask=expert_mask if expert_mask is not None else (emap>=0).to(torch.int32)
        sid,sw,eid,nvalid,bound=native.ep_sorting(ids,weights,mask,352,22,16,emap,expert_layout=layout)
        count=torch.empty(22,dtype=torch.int32,device=a.device)
        c1,c2=shared_scratch(workspace13,22*256,out,avoid)
        aq,asc,q,qs=quant_scratch(workspace2,22*256,a.device,avoid=a)
        aq=aq.view(22,256,4096)
        asc=asc.view(22,256)
        q=q.view(22,256,1536)
        qs=qs.view(22,256)
        c1=c1.view(22,256,3072)
        c2=c2.view(22,256,4096)
        prepare_masked[(22,32)](a.view(torch.uint8),a1_scale,sid,layout,aq.view(torch.uint8),asc,count,index,M=m,num_warps=4)
        expected=max(1,(m*8+351)//352)
        _masked_gemm((aq,asc),(p1,s1.view(22,3072)),c1,count,expected)
        masked_quant[(22,32)](q,qs,c1,count,sw,layout,num_warps=4)
        add=native.shared_add_provider(out) if native.shared_add_provider is not None else None
        _masked_gemm((q,qs),(p2,s2.view(22,4096)),c2,count,expected)
        c2=c2.view(-1,4096)
    else:
        mask=expert_mask if expert_mask is not None else (emap>=0).to(torch.int32)
        sid,sw,eid,nvalid,bound=native.ep_sorting(ids,weights,mask,352,22,256,emap)
        c1,c2=shared_scratch(workspace13,bound,out,avoid)
        aq,asc,q,qs=quant_scratch(workspace2,bound,a.device,avoid=a)
        midx=torch.empty(bound,dtype=torch.int32,device=a.device)
        contiguous_metadata[(triton.cdiv(bound,256),)](sid,nvalid,ids,emap,midx,index,M=m,ROWS=bound,BLOCK=256,num_warps=4)
        native._ep_gather_sorted_kernel[(triton.cdiv(bound,32),)](
            a.view(torch.uint8),a1_scale.view(-1),sid,nvalid,aq.view(torch.uint8),asc,
            m,4096,32,2048,num_warps=4)
        _contiguous_gemm((aq,asc),(p1,s1.view(22,3072)),c1,midx)
        cfg=native.ACT_CFG
        bm=cfg['BLOCK_M']
        native._ep_swiglustep_quant_kernel[(max(1,min(triton.cdiv(bound,bm),native._num_cus(a.device.index)*cfg['CTAS_PER_CU'])),)](
            q,qs,c1,sw,nvalid,D=1536,LIMIT=7.,DTYPE_MAX=448.,BLOCK_M=bm,CH=512,
            num_warps=cfg['num_warps'],num_stages=1)
        add=native.shared_add_provider(out) if native.shared_add_provider is not None else None
        _contiguous_gemm((q,qs),(p2,s2.view(22,4096)),c2,midx)
    add_tensor,add_start=(None,0) if add is None else add
    bn=native.SUM_CFG['BLOCK_N']
    indexed_sum[(m,4096//bn)](c2,index,out,ids,emap,
        out if add_tensor is None else add_tensor,0 if add_tensor is None else add_tensor.stride(0),
        add_start,add_start+(0 if add_tensor is None else add_tensor.shape[0]),
        BLOCK_N=bn,SCALE=float(routed_scaling_factor),HAS_ADD=add_tensor is not None,
        num_warps=native.SUM_CFG['num_warps'])
    return out


def run_packed_experts(a,w1,w2,weights,ids,emap,s1,s2,**kw):
    assert w1.shape==(22,3072,4096) and w2.shape==(22,4096,1536)
    assert int(kw.pop('global_num_experts',352))==352
    m=a.shape[0]
    if m<=native.CHUNK:
        return apply_chunk(a,w1,w2,weights,ids,emap,s1,s2,**kw)
    scale=kw.pop('a1_scale',None)
    out=kw.pop('out',None)
    if out is None:
        out=torch.empty((m,4096),dtype=torch.bfloat16,device=a.device)
    for start in range(0,m,native.CHUNK):
        end=min(m,start+native.CHUNK)
        apply_chunk(a[start:end],w1,w2,weights[start:end],ids[start:end],emap,s1,s2,
                    a1_scale=None if scale is None else scale[start:end],out=out[start:end],
                    preserve_chunk_output=out,**kw)
    return out
