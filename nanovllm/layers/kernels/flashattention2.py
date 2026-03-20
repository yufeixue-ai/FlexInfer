import torch
import triton
import triton.language as tl


@triton.jit
def _attn_inner_loop(
    acc,
    rowmax_i,
    rowsum_i,
    q,
    k_ptrs,
    v_ptrs,
    k_seq_stride,
    v_seq_stride,
    offs_m,
    qk_scale,
    n_size,
    BLOCK_N_SIZE: tl.constexpr,
):

    LOG2E = 1.4426950408889634
     
    for block_n_start in range(0, n_size, BLOCK_N_SIZE):
        block_n_ptrs = block_n_start * k_seq_stride
        block_n_offs = block_n_start + tl.arange(0, BLOCK_N_SIZE)
        k_mask = block_n_offs[:, None] < n_size
        k = tl.load(k_ptrs + block_n_ptrs, mask=k_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k))

        offs_k = block_n_offs
        mask = offs_m[:, None] >= offs_k[None, :]
        qk = qk * (qk_scale * LOG2E) + tl.where(mask, 0, -1.0e8)
        rowmax_ij = tl.maximum(rowmax_i, tl.max(qk, 1))  
        qk -= rowmax_ij[:, None]  

        p = tl.math.exp2(qk)
        rowsum_ij = tl.sum(p, 1)  

        alpha = tl.math.exp2(rowmax_i - rowmax_ij)
        rowsum_i = rowsum_i * alpha + rowsum_ij

        acc = acc * alpha[:, None]

        # compute Out = PV
        v = tl.load(v_ptrs + block_n_ptrs, mask=k_mask, other=0.0)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        rowmax_i = rowmax_ij

    return acc, rowsum_i


@triton.jit
def flashattention2_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    q_batch_stride,
    q_heads_stride,
    q_seq_stride,
    q_dim_stride,
    k_batch_stride,
    k_heads_stride,
    k_seq_stride,
    k_dim_stride,
    v_batch_stride,
    v_heads_stride,
    v_seq_stride,
    v_dim_stride,
    out_batch_stride,
    out_heads_stride,
    out_seq_stride,
    out_dim_stride,
    num_kv_groups,
    n_heads,
    m_size,  # seqlen of query
    n_size,  # seqlen of key/value
    HEAD_DIM: tl.constexpr,
    BLOCK_M_SIZE: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
    qk_scale,
):
    ## 1. determine parallel block ids
    # seqlen of query
    block_m_idx = tl.program_id(0)
    
    # bsz*n_heads
    head_idx = tl.program_id(1)
    cur_batch_idx = head_idx // n_heads
    cur_head_idx = head_idx % n_heads
    cur_kv_head_idx = cur_head_idx // num_kv_groups
    
    ## 2. determine 2d offs of q/k/v/o
    offs_m_range = tl.arange(0, BLOCK_M_SIZE)
    offs_n_range = tl.arange(0, BLOCK_N_SIZE)
    offs_d_range = tl.arange(0, HEAD_DIM)
    
    offs_m = block_m_idx * BLOCK_M_SIZE + offs_m_range
    
    offs_q = (
        cur_batch_idx * q_batch_stride 
        + cur_head_idx * q_heads_stride 
        + (
            offs_m[:, None] * q_seq_stride + offs_d_range[None, :] * q_dim_stride
        )
    )  # real offsets
    offs_k = (
        cur_batch_idx * k_batch_stride
        + cur_kv_head_idx * k_heads_stride
        + (
            offs_n_range[:, None] * k_seq_stride + offs_d_range[None, :] * k_dim_stride
        )
    )  # related offsets
    offs_v = (
        cur_batch_idx * v_batch_stride
        + cur_kv_head_idx * v_heads_stride
        + (
            offs_n_range[:, None] * v_seq_stride + offs_d_range[None, :] * v_dim_stride
        )
    )  # related offsets
    offs_o = (
        cur_batch_idx * out_batch_stride
        + cur_head_idx * out_heads_stride
        + (
            offs_m[:, None] * out_seq_stride + offs_d_range[None, :] * out_dim_stride
        )
    )  # real offsets
    
    q_ptrs = q_ptr + offs_q
    k_ptrs = k_ptr + offs_k
    v_ptrs = v_ptr + offs_v
    out_ptrs = o_ptr + offs_o
    
    q_mask = offs_m[:, None] < m_size
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    acc = tl.zeros([BLOCK_M_SIZE, HEAD_DIM], dtype=tl.float32)
    
    rowmax_i = tl.zeros([BLOCK_M_SIZE], dtype=tl.float32) - float("inf")
    rowsum_i = tl.zeros([BLOCK_M_SIZE], dtype=tl.float32)
    
    acc, rowsum_i = _attn_inner_loop(
        acc,
        rowmax_i,
        rowsum_i,
        q,
        k_ptrs,
        v_ptrs,
        k_seq_stride,
        v_seq_stride,
        offs_m,
        qk_scale,
        n_size,
        BLOCK_N_SIZE,
    )
    
    acc = acc / rowsum_i[:, None]
    out_mask = offs_m[:, None] < m_size
    tl.store(out_ptrs, acc, mask=out_mask)
    
    

@torch.no_grad()
def flashattention2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qk_scale):
    BLOCK_SIZE = 64  
    num_kv_groups = q.shape[1] // k.shape[1]  
    output = torch.empty_like(q)

    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert q.dtype == k.dtype == v.dtype == output.dtype, (
        f"All tensors must have the same dtype: {q.dtype}, {k.dtype}, {v.dtype}, {output.dtype}"
    )

    bs, n_heads, m_size, head_dim = q.size()

    n_size = k.shape[2]

    grid = lambda meta: (triton.cdiv(m_size, BLOCK_SIZE), bs * n_heads, 1) 

    flashattention2_kernel[grid](
        q,
        k,
        v,
        output,
        *q.stride(),  
        *k.stride(),  
        *v.stride(),  
        *output.stride(), 
        num_kv_groups,
        n_heads,
        m_size,
        n_size,
        head_dim,
        BLOCK_SIZE, 
        BLOCK_SIZE,
        qk_scale,
    )
    return output


def _standard_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qk_scale: float):
    bs, n_heads_q, m_size, head_dim = q.shape
    n_heads_k = k.shape[1]
    n_size = k.shape[2]
    num_kv_groups = n_heads_q // n_heads_k
    if num_kv_groups > 1:
        k = k.repeat_interleave(num_kv_groups, dim=1)
        v = v.repeat_interleave(num_kv_groups, dim=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) * qk_scale
    causal_mask = torch.tril(torch.ones(m_size, n_size, device=q.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
    scores = scores.masked_fill(causal_mask == 0, float("-inf"))
    attn_weights = torch.nn.functional.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.matmul(attn_weights, v)
    return out


def test_flash_attention_v2_correctness():
    torch.manual_seed(42)
    device = "cuda"

    bs, n_heads_q, n_heads_k, m_size, n_size, head_dim = 16, 8, 4, 256, 256, 128
    assert m_size == n_size, "Only prefill stage is supported"
    qk_scale = head_dim**-0.5

    q = torch.randn(bs, n_heads_q, m_size, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(bs, n_heads_k, n_size, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(bs, n_heads_k, n_size, head_dim, device=device, dtype=torch.float16)

    out_triton = flashattention2(q, k, v, qk_scale)
    out_ref = _standard_causal_attention(q, k, v, qk_scale)

    assert out_triton.shape == out_ref.shape
    assert torch.allclose(out_triton, out_ref, atol=1e-2, rtol=1e-2), (
        f"Max diff: {torch.max(torch.abs(out_triton - out_ref)).item():.6f}"
    )
    print("TEST SUCCESS!")
    
if __name__ == "__main__":
    test_flash_attention_v2_correctness()
    