"""Isolated exact-window page-table crop for Step5 SWA prefill."""
import triton
import triton.language as tl
import torch

@triton.jit
def _crop_swa_table(SEQ,CUQ,TABLE,OUTSEQ,OUTTABLE,
                    ROWS:tl.constexpr,OLD_COLS:tl.constexpr,NEW_COLS:tl.constexpr,
                    PAGE:tl.constexpr,LEFT:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    end=tl.load(SEQ+row)
    qlen=tl.load(CUQ+row+1)-tl.load(CUQ+row)
    first_page=tl.maximum(end-qlen-LEFT,0)//PAGE
    length=end-first_page*PAGE
    col=tl.arange(0,BLOCK)
    old_col=first_page+col
    live=(col<NEW_COLS)&(old_col<OLD_COLS)&(col<tl.cdiv(length,PAGE))
    block=tl.load(TABLE+row*OLD_COLS+old_col,live,0)
    tl.store(OUTTABLE+row*NEW_COLS+col,block,col<NEW_COLS)
    tl.store(OUTSEQ+row,length)

def crop_sliding_paged_kv(seqused_k,cu_seqlens_q,block_table,
                          max_seqlen_q,max_seqlen_k,window_left,block_size=64):
    if block_size!=64 or window_left<0 or max_seqlen_q<=1:
        raise ValueError("Step5 SWA crop requires prefill, a finite left window and64-token pages")
    if (block_table.ndim!=2 or seqused_k.ndim!=1 or cu_seqlens_q.ndim!=1
            or block_table.shape[0]!=seqused_k.shape[0]
            or cu_seqlens_q.shape[0]!=seqused_k.shape[0]+1
            or block_table.stride(1)!=1
            or block_table.stride(0)!=block_table.shape[1]):
        raise ValueError("Invalid Step5 SWA table geometry")
    # Round down the retained starting position to a physical page boundary.
    # The extra <=63 tokens remain masked by the unchanged native SWA window.
    bound=min(max_seqlen_k,max_seqlen_q+window_left+block_size-1)
    columns=min(block_table.shape[1],triton.cdiv(bound,block_size))
    cropped=torch.empty((block_table.shape[0],columns),device=block_table.device,
                        dtype=block_table.dtype)
    lengths=torch.empty_like(seqused_k)
    _crop_swa_table[(block_table.shape[0],)](
        seqused_k,cu_seqlens_q,block_table,lengths,cropped,
        block_table.shape[0],block_table.shape[1],columns,
        block_size,window_left,triton.next_power_of_2(columns))
    return lengths,cropped,bound
