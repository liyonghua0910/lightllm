import torch

import triton
import triton.language as tl
import math


@triton.jit
def _fwd_kernel_token_att1(
    Q, K, sm_scale, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    att_stride_h, att_stride_bs,
    kv_group_num,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)
    
    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    cur_batch_start_index = 0
    cur_batch_end_index = cur_batch_seq_len

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)

    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + off_q + start_mark)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * offs_n_new, 
                        mask=offs_n_new < cur_batch_end_index, other=0)
        off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        off_o = cur_head * att_stride_h + (cur_batch_in_all_start_index + offs_n) * att_stride_bs
        tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)
    return


@torch.no_grad()
def token_att_fwd(q, k, att_out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, max_len_in_batch):
    BLOCK = 32
    # shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    sm_scale = 1.0 / (Lk ** 0.5)

    batch, head_num = B_req_idx.shape[0], q.shape[1]

    grid = (batch, head_num, triton.cdiv(max_len_in_batch, BLOCK))
    kv_group_num = q.shape[1] // k.shape[1]
    
    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2

    _fwd_kernel_token_att1[grid](
        q, k, sm_scale, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen,
        att_out,
        Req_to_tokens.stride(0), Req_to_tokens.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        att_out.stride(0), att_out.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _fwd_kernel_token_att1_h2o(
    Q, K, sm_scale, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b, stride_req_to_tokens_h, stride_req_to_tokens_s,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    att_stride_h, att_stride_bs,
    kv_group_num,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)
    
    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    cur_batch_start_index = 0
    cur_batch_end_index = cur_batch_seq_len

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)

    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + off_q + start_mark)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx 
                                      + stride_req_to_tokens_h * cur_kv_head
                                      + stride_req_to_tokens_s * offs_n_new, 
                        mask=offs_n_new < cur_batch_end_index, other=0)
        off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        off_o = cur_head * att_stride_h + (cur_batch_in_all_start_index + offs_n) * att_stride_bs
        tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)
    return


@torch.no_grad()
def token_att_fwd_h2o(q, k, att_out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, max_len_in_batch):
    BLOCK = 32
    # shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    sm_scale = 1.0 / (Lk ** 0.5)

    batch, head_num = B_req_idx.shape[0], q.shape[1]

    grid = (batch, head_num, triton.cdiv(max_len_in_batch, BLOCK))
    kv_group_num = q.shape[1] // k.shape[1]
    
    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2

    _fwd_kernel_token_att1_h2o[grid](
        q, k, sm_scale, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen,
        att_out,
        Req_to_tokens.stride(0), Req_to_tokens.stride(1), Req_to_tokens.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        att_out.stride(0), att_out.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _fwd_kernel_token_att1_int8(
    Q, K, K_scale, sm_scale, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_ksbs, stride_ksh, stride_ksd,
    att_stride_h, att_stride_bs,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    cur_batch_start_index = 0
    cur_batch_end_index = cur_batch_seq_len

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)

    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + off_q + start_mark)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)
        off_k = k_loc[:, None] * stride_kbs + cur_head * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        off_ks = k_loc[:, None] * stride_ksbs + cur_head * stride_ksh
        k_scale = tl.load(K_scale + off_ks, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        att_value = tl.sum(q[None, :] * k * k_scale, 1)
        att_value *= sm_scale
        off_o = cur_head * att_stride_h + (cur_batch_in_all_start_index + offs_n) * att_stride_bs
        tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)
    return


@torch.no_grad()
def token_att_fwd_int8k(q, k, k_scale, att_out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, max_input_len):
    BLOCK = 32
    # shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    sm_scale = 1.0 / (Lk ** 0.5)

    batch, head_num = B_req_idx.shape[0], q.shape[1]

    grid = (batch, head_num, triton.cdiv(max_input_len, BLOCK))

    num_warps = 4 if Lk <= 64 else 8
    num_warps = 2

    _fwd_kernel_token_att1_int8[grid](
        q, k, k_scale, sm_scale, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen,
        att_out,
        Req_to_tokens.stride(0), Req_to_tokens.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        att_out.stride(0), att_out.stride(1),
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return


def test():
    batch_size = 1
    head_num = 10
    head_dim = 128
    total_cache_size = 1000
    seq_cache_size = 65
    max_seq_num = 100
    torch.manual_seed(0)
    q = 0.5 - torch.rand(batch_size, head_num, head_dim, device='cuda', dtype=torch.float16)
    k = 2 + 2 * torch.rand(total_cache_size, head_num, head_dim, device='cuda', dtype=torch.float16)
    att_out = torch.zeros(head_num, seq_cache_size * batch_size, device='cuda', dtype=torch.float16)
    Req_to_tokens = torch.randint(0, total_cache_size, [max_seq_num, head_num, seq_cache_size], device='cuda', dtype=torch.int32)
    B_req_idx = torch.randint(0, max_seq_num, [batch_size], device='cuda', dtype=torch.int32)
    B_Seqlen = torch.ones(batch_size, device='cuda', dtype=torch.int32) * seq_cache_size
    B_Start_Loc = torch.cumsum(B_Seqlen, 0) - B_Seqlen
    max_len_in_batch = torch.max(B_Seqlen).item()
    
    token_att_fwd_h2o(q, k, att_out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, max_len_in_batch)
    
    batch_idxs = Req_to_tokens[B_req_idx.long()]
    torch_out = torch.zeros(batch_size, head_num, seq_cache_size, device='cuda', dtype=torch.float16)
    for b, req_idxs in enumerate(batch_idxs):
        for h, head_idxs in enumerate(req_idxs):
            head_q = q[b, h].unsqueeze(0)
            head_k = k[head_idxs.long(), h]
            torch_out[b, h] = torch.mm(head_q, head_k.t()) / head_dim ** 0.5
    torch_out = torch_out.transpose(0,1).view(head_num, -1)
    print(torch.norm(torch_out-att_out))

if __name__ == '__main__':
    test()