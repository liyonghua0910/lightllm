import torch

import triton
import triton.language as tl
import math
import torch.nn.functional as F

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, B_Start_Loc, B_Seqlen,  # B_LOC 内部记录每个batch 输入的真实位置， B_SEQ_len 记录当前输入的真实长度
    ProbCur, ProbCum, Out,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_pbs, stride_ph, stride_pb, stride_pt,
    stride_obs, stride_oh, stride_od,
    kv_group_num,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd

    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
                    mask=(start_n + offs_n[None, :]) < cur_batch_seq_len, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # 得到真实的 m_i 和 l_i 之后，重新遍历一轮 key tokens，计算累计注意力概率
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
                    mask=(start_n + offs_n[None, :]) < cur_batch_seq_len, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        p_full = tl.exp(qk - m_i[:, None]) / l_i[:, None]
        p_full = tl.where((offs_m[:, None] < cur_batch_seq_len), p_full, 0)
        p_cum = tl.sum(p_full, 0)
        tl.store(pointer=ProbCum + cur_batch * stride_pbs + cur_head * stride_ph + start_m * stride_pb + (start_n + offs_n) * stride_pt, 
                 value=p_cum, mask=start_n + offs_n < cur_batch_seq_len)
        p_cur = tl.where(offs_m[:, None] == cur_batch_seq_len - 1, p_full, 0)
        p_cur = tl.sum(p_cur, 0)
        tl.store(pointer=ProbCur + cur_batch * stride_pbs + cur_head * stride_ph + start_m * stride_pb + (start_n + offs_n) * stride_pt, 
                 value=p_cur, mask=start_n + offs_n < cur_batch_seq_len)
        
    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
    return


@torch.no_grad()
def context_attention_fwd(q, k, v, cur_score, cum_score, o, b_start_loc, b_seq_len, max_input_len):
    TESLA = True
    BLOCK = 128 if not TESLA else 64
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}

    sm_scale = 1.0 / (Lq ** 0.5)  # 计算scale系数
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    m_blocks = triton.cdiv(max_input_len, BLOCK)
    grid = (batch, head, m_blocks)  # batch, head,

    p_cur = torch.zeros([batch, head, m_blocks, max_input_len], dtype=torch.float16, device=q.device)
    p_cum = torch.zeros([batch, head, m_blocks, max_input_len], dtype=torch.float16, device=q.device)
    num_warps = 4 if Lk <= 64 else 8
    _fwd_kernel[grid](
        q, k, v, sm_scale, b_start_loc, b_seq_len, 
        p_cur, p_cum, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        p_cum.stride(0), p_cum.stride(1), p_cum.stride(2), p_cum.stride(3),
        o.stride(0), o.stride(1), o.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    cur_score[...] = p_cur.sum(dim=2)
    cum_score[...] = p_cum.sum(dim=2)


def torch_att(q, k, v, b_start_loc, b_seq_len, max_input_len):
    o = torch.empty_like(q)
    _, q_head_num, head_dim = q.shape
    kv_group_num = q.shape[1] // k.shape[1]
    batch_size = len(b_seq_len)
    cum_score = torch.zeros((batch_size, q_head_num, max_input_len), dtype=torch.float16, device=q.device)
    cur_score = torch.zeros((batch_size, q_head_num, max_input_len), dtype=torch.float16, device=q.device)
    for i in range(batch_size):
        seq_len = b_seq_len[i]
        start_loc = b_start_loc[i]
        q_i = q[start_loc:start_loc+seq_len]
        k_i = k[start_loc:start_loc+seq_len, :, None, :].expand(-1, -1, kv_group_num, -1).reshape(seq_len, q_head_num, head_dim)
        v_i = v[start_loc:start_loc+seq_len, :, None, :].expand(-1, -1, kv_group_num, -1).reshape(seq_len, q_head_num, head_dim)
        s_i = torch.matmul(q_i.permute(1,0,2), k_i.permute(1,2,0)) / torch.sqrt(torch.tensor(head_dim))
        mask = torch.zeros_like(s_i)
        indx = torch.triu_indices(seq_len, seq_len, offset=1)
        mask[:, indx[0], indx[1]] = -torch.inf
        p_i = torch.softmax(s_i + mask, dim=-1)
        cur_score[i, :, :seq_len] = p_i[:,-1,:]
        cum_score[i, :, :seq_len] = p_i.sum(1)
        o_i = torch.matmul(p_i, v_i.permute(1,0,2))
        o[start_loc:start_loc+seq_len] = o_i.permute(1,0,2)
    return cur_score, cum_score, o


def test():
    import torch
    torch.manual_seed(37)

    requests = [100, 200, 300, 400]
    batch_size = len(requests)
    head_num = 16
    head_dim = 128

    device = "cuda" if torch.cuda.is_available() else "cpu"
    b_seq_len = torch.tensor(requests, dtype=torch.int32, device=device)
    b_start_loc = b_seq_len.cumsum(-1) - b_seq_len
    max_input_len = b_seq_len.max().item()
    batch_total_tokens = b_seq_len.sum().item()

    q = torch.empty((batch_total_tokens, head_num, head_dim), dtype=torch.float16, device=device).normal_(mean=0.2, std=0.2)
    k = torch.empty((batch_total_tokens, head_num, head_dim), dtype=torch.float16, device=device).normal_(mean=0.2, std=0.2)
    v = torch.empty((batch_total_tokens, head_num, head_dim), dtype=torch.float16, device=device).normal_(mean=0.5, std=1.0)
    o = torch.empty((batch_total_tokens, head_num, head_dim), dtype=torch.float16, device=device).normal_(mean=0.2, std=0.2)

    ### Torch Kernel
    torch_cur_score, torch_cum_score, torch_out = torch_att(q, k, v, b_start_loc, b_seq_len, max_input_len)

    ### Triton Kernel
    triton_cur_score = torch.zeros([batch_size, head_num, max_input_len], dtype=torch.float16, device=device)
    triton_cum_score = torch.zeros([batch_size, head_num, max_input_len], dtype=torch.float16, device=device)
    context_attention_fwd(q, k, v, triton_cur_score, triton_cum_score, o, b_start_loc, b_seq_len, max_input_len)
    triton_out = o

    ### Comparison
    isclose = torch.isclose(torch_cur_score, triton_cur_score, atol=0, rtol=0.01)
    if isclose.sum().item() / isclose.numel() > 0.99:
        print("torch_cur_score & triton_cur_score difference test passed!")

    isclose = torch.isclose(torch_cum_score, triton_cum_score, atol=0, rtol=0.01)
    if isclose.sum().item() / isclose.numel() > 0.99:
        print("torch_cum_score & triton_cum_score difference test passed!")
    
    isclose = torch.isclose(torch_out, triton_out, atol=0, rtol=0.01)
    if isclose.sum().item() / isclose.numel() > 0.99:
        print("torch_out & triton_out difference test passed!")

if __name__ == '__main__':
    test()