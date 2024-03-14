def init_req_to_token_indexes(req_to_token_indexs, b_req_idx, b_seq_len, max_len_in_batch, alloc_mem_index):
    start_index = 0
    b_seq_len_numpy = b_seq_len.cpu().numpy()
    b_req_idx_numpy = b_req_idx.cpu().numpy()
    for i in range(len(b_seq_len)):
        cur_seq_len = b_seq_len_numpy[i]
        req_to_token_indexs[b_req_idx_numpy[i], 0:cur_seq_len] = alloc_mem_index[start_index:start_index + cur_seq_len]
        start_index += cur_seq_len
    return

def init_req_to_atten_indexes(req_to_atten_indexs, req_to_atten_scores, b_req_idx, b_seq_len, max_len_in_batch, alloc_mem_index):
    start_index = 0
    for i in range(len(b_seq_len)):
        cur_seq_len = b_seq_len[i]
        req_to_atten_indexs[b_req_idx[i], :, :, 0:cur_seq_len] = alloc_mem_index[:, :, start_index:start_index+cur_seq_len]
        start_index += cur_seq_len
    req_to_atten_scores[b_req_idx.long()] = 0
    return
