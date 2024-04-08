import logging
import os
import torch
import torch.distributed as dist
from ..transformer_layer_infer import TransformerLayerInfer
from ...infer_struct import InferStateInfo
from ...splitfuse_infer_struct import SplitFuseInferStateInfo
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv, destindex_copy_kv_finegrained
from typing import Tuple

# import debugpy; debugpy.connect(('10.119.25.81', 5678))

from lightllm.utils.log_utils import init_logger
logger = init_logger(__name__)


class TransformerLayerInferTpl(TransformerLayerInfer):
    """
    """
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        # need to set by subclass
        self.eps_ = 1e-5 
        self.tp_q_head_num_ = -1
        self.tp_k_head_num_ = -1
        self.tp_v_head_num_ = -1
        self.tp_o_head_num_ = -1
        self.head_dim_ = -1
        self.embed_dim_ = -1
        return
    
    def _att_norm(self, input, infer_state:InferStateInfo, layer_weight)->torch.Tensor:
        raise Exception("need to impl")
    
    def _ffn_norm(self, input, infer_state:InferStateInfo, layer_weight)->torch.Tensor:
        raise Exception("need to impl")
    
    def _pre_cache_kv(self, infer_state:InferStateInfo, layer_weight)->Tuple[torch.Tensor, torch.Tensor]:
        if os.getenv('ENABLE_CACHE_DROPPING') == '1':
            cache_k = infer_state.key_buffer
            cache_v = infer_state.value_buffer
        elif infer_state.mem_is_contiguous:
            cache_k = infer_state.mem_manager.key_buffer[self.layer_num_][infer_state.mem_start:infer_state.mem_end, :, :]
            cache_v = infer_state.mem_manager.value_buffer[self.layer_num_][infer_state.mem_start:infer_state.mem_end, :, :]
        else:
            cache_k = infer_state.key_buffer
            cache_v = infer_state.value_buffer 
        return cache_k, cache_v

    def _get_qkv(self, input, cache_k, cache_v, infer_state:InferStateInfo, layer_weight)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise Exception("need to impl")
    
    def _post_cache_kv(self, cache_k, cache_v, infer_state:InferStateInfo, layer_weight):
        mem_manager = infer_state.mem_manager
        if os.getenv('ENABLE_CACHE_DROPPING') == '1':
            self._copy_kv_to_finegrained_mem_cache(cache_k, cache_v, infer_state.finegrained_mem_index, mem_manager)
        elif not infer_state.mem_is_contiguous:
            self._copy_kv_to_mem_cache(cache_k, cache_v, infer_state.mem_index, mem_manager)

    def _copy_kv_to_mem_cache(self, key_buffer, value_buffer, mem_index, mem_manager):
        destindex_copy_kv(key_buffer, mem_index, mem_manager.key_buffer[self.layer_num_])
        destindex_copy_kv(value_buffer, mem_index, mem_manager.value_buffer[self.layer_num_])
    
    def _copy_kv_to_finegrained_mem_cache(self, key_buffer, value_buffer, finegrained_mem_index, mem_manager):
        """ Move keys/values at selected token positions from temporary buffer to kv cache according to dest index.
            Shapes:
                key_buffer/value_buffer: (token_num, head_num, head_dim)
                finegrained_mem_index: (layer_num, head_num, token_num)
                mem_manager.key_buffer/value_buffer: [layer_num * (size, head_num, head_dim)]
            Pseudocode:
                L ← current layer index
                N ← total_token_num
                H ← head_num
                for i = 0,1,...,N-1:
                    for j = 0,1,...,H-1:
                        p ← finegrained_mem_index[L,j,i]
                        mem_manager.key_buffer[p,j,:] ← key_buffer[i,j,:]
        """
        destindex_copy_kv_finegrained(key_buffer, finegrained_mem_index[self.layer_num_], mem_manager.key_buffer[self.layer_num_])
        destindex_copy_kv_finegrained(value_buffer, finegrained_mem_index[self.layer_num_], mem_manager.value_buffer[self.layer_num_])
        return

    def _context_attention_kernel(self, q, k, v, infer_state:InferStateInfo, layer_weight, out=None, return_att_score=False)->torch.Tensor:
        raise Exception("need to impl")
    
    def _token_attention_kernel(self, q, infer_state:InferStateInfo, layer_weight, out=None, return_att_score=False)->torch.Tensor:
        raise Exception("need to impl")
    
    def _splitfuse_attention_kernel(self, q, infer_state:SplitFuseInferStateInfo, layer_weight, out=None)->torch.Tensor:
        raise Exception("need to impl")

    def _get_o(self, input, infer_state:InferStateInfo, layer_weight)->torch.Tensor:
        raise Exception("need to impl")

    def _ffn(self, input, infer_state:InferStateInfo, layer_weight)->torch.Tensor:
        raise Exception("need to impl")

    def _drop_cache(self, att_score, cache_k, cache_v, infer_state: InferStateInfo):

        batch_size, head_num, _ = att_score.shape
        req_to_atten_indexs = infer_state.req_manager.req_to_atten_indexs
        req_to_atten_scores = infer_state.req_manager.req_to_atten_scores
        req_to_atten_times = infer_state.req_manager.req_to_atten_times

        cache_size = infer_state.req_manager.layers_cache_size[self.layer_num_]
        sink_size = infer_state.req_manager.layers_cache_sink_size[self.layer_num_]
        top_size = infer_state.req_manager.layers_cache_top_size[self.layer_num_]
        local_size = infer_state.req_manager.layers_cache_local_size[self.layer_num_]

        if infer_state.is_prefill:
            ######## BATCH PROCESSING CODE (WITH UNKNOWN BUGS) ############################################
            # ## 批量处理不需要丢弃词的序列
            # cached_kv_pos = [None for _ in range(batch_size)]
            # no_evict_req_ids = torch.nonzero(infer_state.b_att_len[self.layer_num_] <= cache_size).squeeze(-1)
            # no_evict_req_num = no_evict_req_ids.numel()
            # if no_evict_req_num > 0:
            #     no_evict_req_idxs = infer_state.b_req_idx[no_evict_req_ids].long()
            #     no_evict_att_len = infer_state.b_att_len[self.layer_num_][no_evict_req_ids]
            #     no_evict_start_loc = infer_state.b_start_loc[no_evict_req_ids]
            #     no_evict_max_att_len = no_evict_att_len.max().item()
            #     no_evict_atten_score = att_score[no_evict_req_ids, :, :no_evict_max_att_len]
            #     #### 用下面的方式批量初始化no_evict_atten_times会导致奇怪的bug，所以逐个初始化
            #     # no_evict_atten_times = torch.arange(0, -no_evict_max_att_len, -1, dtype=torch.int32, device='cuda').repeat(no_evict_req_num, head_num, 1) + no_evict_att_len.unsqueeze(-1).unsqueeze(-1)
            #     no_evict_atten_times = torch.zeros_like(no_evict_atten_score, dtype=torch.int32)
            #     for i, j in enumerate(no_evict_att_len):
            #         no_evict_atten_times[i, :, :j] = torch.arange(j, 0, -1)

            #     ## 填充各token的累积注意力分数、各token被关注的次数
            #     req_to_atten_scores[self.layer_num_][no_evict_req_idxs, :, :no_evict_max_att_len] = no_evict_atten_score
            #     req_to_atten_times[self.layer_num_][no_evict_req_idxs, :, :no_evict_max_att_len] = no_evict_atten_times

            #     ## 收集需要缓存的kv在buffer中的位置
            #     for i, j in enumerate(no_evict_req_ids):
            #         cached_kv_pos[j] = no_evict_start_loc[i] + torch.arange(0, no_evict_att_len[i].item(), device='cuda').repeat(head_num, 1)
            
            # ## 批量处理需要丢弃词的序列
            # to_evict_req_ids = torch.nonzero(infer_state.b_att_len[self.layer_num_] > cache_size).squeeze(-1)
            # to_evict_req_num = to_evict_req_ids.numel()
            # if to_evict_req_num > 0:
            #     to_evict_req_idxs = infer_state.b_req_idx[to_evict_req_ids].long()
            #     to_evict_att_len = infer_state.b_att_len[self.layer_num_][to_evict_req_ids]
            #     to_evict_start_loc = infer_state.b_start_loc[to_evict_req_ids]
            #     to_evict_max_att_len = to_evict_att_len.max().item()
            #     to_evict_atten_score = att_score[to_evict_req_ids, :, :to_evict_max_att_len]
            #     to_evict_atten_times = torch.arange(0, -to_evict_max_att_len, -1, dtype=torch.int32, device='cuda').repeat(to_evict_req_num, head_num, 1) + to_evict_att_len.unsqueeze(-1).unsqueeze(-1)

            #     ## 构建topk候选词的掩码: 在序列有效长度以内、非sink、非local
            #     candidate_mask = torch.zeros_like(to_evict_atten_score)
            #     for i in range(to_evict_req_num):
            #         candidate_mask[i, :, :to_evict_att_len[i]] = 1
            #         candidate_mask[i, :, :sink_size] = 0
            #         candidate_mask[i, :, to_evict_att_len[i]-local_size:to_evict_att_len[i]] = 0
            #     assert candidate_mask.sum() == head_num * (to_evict_att_len - sink_size - local_size).sum()

            #     ## 排序选出topk在序列中的下标，并直接生成sink和local在序列中的下标
            #     to_evict_redist_score = torch.where(candidate_mask == 1, to_evict_atten_score / to_evict_atten_times, -torch.inf)
            #     b_h_top_pos = torch.argsort(to_evict_redist_score, descending=True, stable=True)[:,:,:top_size]
            #     b_h_sink_pos = torch.arange(0, sink_size, device='cuda').repeat(to_evict_req_num, head_num, 1)
            #     b_h_local_pos = torch.arange(-1, -local_size-1, -1, device='cuda').repeat(to_evict_req_num, head_num, 1) + to_evict_att_len.unsqueeze(-1).unsqueeze(-1)
            #     b_h_selected_pos = torch.concat((b_h_sink_pos, b_h_top_pos, b_h_local_pos), dim=-1)
            #     b_h_selected_pos = torch.sort(b_h_selected_pos, descending=False).values

            #     ## 填充各token的累积注意力分数、填充各token被关注的次数
            #     idx_b = torch.arange(to_evict_req_num).unsqueeze(-1).unsqueeze(-1).expand_as(b_h_selected_pos)
            #     idx_h = torch.arange(head_num).unsqueeze(0).unsqueeze(-1).expand_as(b_h_selected_pos)
            #     req_to_atten_scores[self.layer_num_][to_evict_req_idxs, :, :cache_size] = to_evict_atten_score[idx_b, idx_h, b_h_selected_pos]
            #     req_to_atten_times[self.layer_num_][to_evict_req_idxs, :, :cache_size] = to_evict_atten_times[idx_b, idx_h, b_h_selected_pos]

            #     ## 收集需要缓存的kv在buffer中的位置
            #     for i, j in enumerate(to_evict_req_ids):
            #         cached_kv_pos[j] = to_evict_start_loc[i] + b_h_selected_pos[i]
            
            # ## 取出真正需要缓存的kv
            # cached_kv_pos = torch.concat(cached_kv_pos, dim=-1)
            # dropped_cache_k = cache_k[cached_kv_pos, torch.arange(self.tp_k_head_num_).repeat(cached_kv_pos.shape[1], 1).t()].transpose(0,1).contiguous()
            # dropped_cache_v = cache_v[cached_kv_pos, torch.arange(self.tp_v_head_num_).repeat(cached_kv_pos.shape[1], 1).t()].transpose(0,1).contiguous()
            # return dropped_cache_k, dropped_cache_v

            ###############################################################################################
            
            ######## ITERATIVE PROCESSING CODE ############################################################
            cached_kv_pos = [None for _ in range(batch_size)]
            for i in range(batch_size):
                req_idx = infer_state.b_req_idx[i]
                seq_len = infer_state.b_seq_len[i]
                att_len = infer_state.b_att_len[self.layer_num_][i]
                start_loc = infer_state.b_start_loc[i]
                atten_idxs = req_to_atten_indexs[self.layer_num_][req_idx, :, :att_len].clone()
                accum_score = att_score[i, :, :att_len].clone()
                assert att_len == seq_len
                if att_len <= cache_size:
                    ## 填充累积注意力分数
                    accum_times = torch.arange(att_len,0,-1).cuda()
                    req_to_atten_scores[self.layer_num_][req_idx, :, :att_len] = accum_score
                    req_to_atten_times[self.layer_num_][req_idx, :, :att_len] = accum_times
                    cached_kv_pos[i] = start_loc + torch.arange(0, att_len, device='cuda').repeat(head_num, 1)
                else:
                    ## 如果缓存预算不够, 那么缓存由sink+top+local三部分组成
                    accum_times = torch.arange(att_len, 0, -1).cuda()
                    sorted_pos = sink_size + torch.argsort((accum_score / accum_times)[:, sink_size:att_len-local_size], descending=True, stable=True)
                    top_pos, evicted_pos = torch.split(sorted_pos, [top_size, att_len-cache_size], dim=1)
                    sink_pos = torch.arange(0, sink_size, device='cuda').repeat(head_num, 1)  # 生成sink的下标
                    local_pos = torch.arange(att_len-local_size, att_len, device='cuda').repeat(head_num,1)  # 生成local的下标
                    selected_pos = torch.concat((sink_pos, top_pos, local_pos), dim=1)
                    selected_pos = torch.sort(selected_pos, descending=False).values
                    assert selected_pos.shape == (head_num, cache_size)
                    cached_kv_pos[i] = start_loc + selected_pos

                    ## 填充累积注意力分数
                    head_index_helper = torch.arange(selected_pos.shape[0], device='cuda').repeat(selected_pos.shape[1], 1).t()
                    req_to_atten_scores[self.layer_num_][req_idx, :, :cache_size] = accum_score[head_index_helper, selected_pos]
                    req_to_atten_times[self.layer_num_][req_idx, :, :cache_size] = accum_times[selected_pos]
    
            cached_kv_pos = torch.concat(cached_kv_pos, dim=-1)
            dropped_cache_k = cache_k[cached_kv_pos, torch.arange(self.tp_k_head_num_).repeat(cached_kv_pos.shape[1], 1).t()].transpose(0,1).contiguous()
            dropped_cache_v = cache_v[cached_kv_pos, torch.arange(self.tp_v_head_num_).repeat(cached_kv_pos.shape[1], 1).t()].transpose(0,1).contiguous()
            return dropped_cache_k, dropped_cache_v
            # ###############################################################################################


        else:
            ######## BATCH PROCESSING CODE ################################################################
            # ## 批量处理不需要丢弃词的序列
            # no_evict_req_ids = torch.nonzero(infer_state.b_att_len[self.layer_num_] <= cache_size).squeeze(-1)
            # no_evict_req_num = no_evict_req_ids.numel()
            # if no_evict_req_num > 0:
            #     no_evict_req_idxs = infer_state.b_req_idx[no_evict_req_ids].long()
            #     no_evict_att_len = infer_state.b_att_len[self.layer_num_][no_evict_req_ids]
            #     no_evict_start_loc = infer_state.b_start_loc[no_evict_req_ids]
            #     no_evict_max_att_len = no_evict_att_len.max().item()
            #     no_evict_atten_score = att_score[no_evict_req_ids, :, :no_evict_max_att_len].clone()
            #     no_evict_atten_score[:,:,:-1] += req_to_atten_scores[self.layer_num_][no_evict_req_idxs, :, :no_evict_max_att_len-1]
            #     # no_evict_atten_times = torch.arange(0, -no_evict_max_att_len, -1, dtype=torch.int32, device='cuda').repeat(no_evict_req_num, head_num, 1) + no_evict_att_len.unsqueeze(-1).unsqueeze(-1)
            #     no_evict_atten_times = torch.zeros_like(no_evict_atten_score, dtype=torch.int32)
            #     for i, j in enumerate(no_evict_att_len):
            #         no_evict_atten_times[i, :, :j] = torch.arange(j, 0, -1)

            #     ## 填充各token的累积注意力分数、填充各token被关注的次数
            #     req_to_atten_scores[self.layer_num_][no_evict_req_idxs, :, :no_evict_max_att_len] = no_evict_atten_score
            #     req_to_atten_times[self.layer_num_][no_evict_req_idxs, :, :no_evict_max_att_len] = no_evict_atten_times

            # ## 批量处理需要丢弃词的序列
            # to_evict_req_ids = torch.nonzero(infer_state.b_att_len[self.layer_num_] > cache_size).squeeze(-1)
            # to_evict_req_num = to_evict_req_ids.numel()
            # if to_evict_req_num > 0:
            #     to_evict_req_idxs = infer_state.b_req_idx[to_evict_req_ids].long()
            #     to_evict_att_len = infer_state.b_att_len[self.layer_num_][to_evict_req_ids]
            #     to_evict_start_loc = infer_state.b_start_loc[to_evict_req_ids]
            #     to_evict_max_att_len = to_evict_att_len.max().item()
            #     to_evict_atten_indexs = req_to_atten_indexs[self.layer_num_][to_evict_req_ids, :, :to_evict_max_att_len]
            #     to_evict_atten_score = att_score[to_evict_req_ids, :, :to_evict_max_att_len].clone()
            #     to_evict_atten_score[:,:,:-1] += req_to_atten_scores[self.layer_num_][to_evict_req_idxs, :, :to_evict_max_att_len-1]
            #     to_evict_atten_times = req_to_atten_times[self.layer_num_][to_evict_req_idxs, :, :to_evict_max_att_len] + 1
            #     assert (to_evict_att_len == cache_size + 1).all()

            #     ## 构建topk候选词的掩码: 在序列有效长度以内、非sink、非local
            #     candidate_mask = torch.zeros_like(to_evict_atten_score)
            #     for i in range(to_evict_req_num):
            #         candidate_mask[i, :, :to_evict_att_len[i]] = 1
            #         candidate_mask[i, :, :sink_size] = 0
            #         candidate_mask[i, :, to_evict_att_len[i]-local_size:to_evict_att_len[i]] = 0
            #     assert candidate_mask.sum() == head_num * (to_evict_att_len - sink_size - local_size).sum()

            #     ## 排序选出topk在序列中的下标，并直接生成sink和local在序列中的下标
            #     to_evict_redist_score = torch.where(candidate_mask == 1, to_evict_atten_score / to_evict_atten_times, -torch.inf)
            #     b_h_top_pos = torch.argsort(to_evict_redist_score, descending=True, stable=True)[:,:,:top_size]
            #     b_h_sink_pos = torch.arange(0, sink_size, device='cuda').repeat(to_evict_req_num, head_num, 1)
            #     b_h_local_pos = torch.arange(-1, -local_size-1, -1, device='cuda').repeat(to_evict_req_num, head_num, 1) + to_evict_att_len.unsqueeze(-1).unsqueeze(-1)
            #     b_h_selected_pos = torch.concat((b_h_sink_pos, b_h_top_pos, b_h_local_pos), dim=-1)
            #     b_h_selected_pos = torch.sort(b_h_selected_pos, descending=False).values
                
            #     ## 在缓存索引映射表更新之前，确定需要被释放的token的索引，并释放对应的缓存
            #     selected_mask = torch.zeros_like(to_evict_atten_indexs)
            #     idx_b = torch.arange(to_evict_req_num).unsqueeze(-1).unsqueeze(-1).expand_as(b_h_selected_pos)
            #     idx_h = torch.arange(head_num).unsqueeze(0).unsqueeze(-1).expand_as(b_h_selected_pos)
            #     selected_mask[idx_b, idx_h, b_h_selected_pos] = 1
            #     batch_head_pos = torch.where(selected_mask==0)
            #     evicted_idx = to_evict_atten_indexs[batch_head_pos]
            #     free_idx = torch.stack((torch.ones_like(evicted_idx)*self.layer_num_, batch_head_pos[1], evicted_idx)).t()
            #     infer_state.mem_manager.free_finegrained_by_index(free_idx)

            #     ## 更新缓存索引映射表、更新各token的累积注意力分数、更新各token被关注的次数
            #     idx_b = torch.arange(to_evict_req_num).unsqueeze(-1).unsqueeze(-1).expand_as(b_h_selected_pos)
            #     idx_h = torch.arange(head_num).unsqueeze(0).unsqueeze(-1).expand_as(b_h_selected_pos)
            #     req_to_atten_indexs[self.layer_num_][to_evict_req_idxs, :, :cache_size] = to_evict_atten_indexs[idx_b, idx_h, b_h_selected_pos]
            #     req_to_atten_scores[self.layer_num_][to_evict_req_idxs, :, :cache_size] = to_evict_atten_score[idx_b, idx_h, b_h_selected_pos]
            #     req_to_atten_times[self.layer_num_][to_evict_req_idxs, :, :cache_size] = to_evict_atten_times[idx_b, idx_h, b_h_selected_pos]
            ###############################################################################################

            ######## ITERATIVE PROCESSING CODE ############################################################
            for i in range(batch_size):
                req_idx = infer_state.b_req_idx[i]
                seq_len = infer_state.b_seq_len[i]
                att_len = infer_state.b_att_len[self.layer_num_][i]
                start_loc = infer_state.b_start_loc[i]
                atten_idxs = req_to_atten_indexs[self.layer_num_][req_idx, :, :att_len].clone()
                accum_score = att_score[i, :, :att_len].clone()
                assert att_len == min(seq_len, cache_size+1)
                accum_score[:, :att_len-1] += req_to_atten_scores[self.layer_num_][req_idx, :, :att_len-1]
                if att_len <= cache_size:
                    accum_times = torch.arange(att_len,0,-1).cuda()
                    req_to_atten_scores[self.layer_num_][req_idx, :, :att_len] = accum_score
                    req_to_atten_times[self.layer_num_][req_idx, :, :att_len] = accum_times
                else:
                    assert att_len == cache_size + 1
                    ## 如果缓存预算达到限制, 那么缓存由sink+top+local三部分组成
                    accum_times = req_to_atten_times[self.layer_num_][req_idx, :, :att_len] + 1
                    sorted_pos = sink_size + torch.argsort((accum_score / accum_times)[:, sink_size:att_len-local_size], descending=True, stable=True)
                    top_pos, evicted_pos = torch.split(sorted_pos, [top_size, att_len-cache_size], dim=1)
                    sink_pos = torch.arange(0, sink_size, device='cuda').repeat(head_num, 1)  # 生成sink的下标
                    local_pos = torch.arange(att_len-local_size, att_len, device='cuda').repeat(head_num,1)  # 生成local的下标
                    selected_pos = torch.concat((sink_pos, top_pos, local_pos), dim=1)
                    selected_pos = torch.sort(selected_pos, descending=False).values
                    assert selected_pos.shape == (head_num, cache_size)

                    ## 在缓存索引映射表更新之前，确定需要被释放的token的索引
                    layer_index_helper = torch.ones_like(evicted_pos, device='cuda') * self.layer_num_
                    head_index_helper = torch.arange(evicted_pos.shape[0], device='cuda').repeat(evicted_pos.shape[1], 1).t()
                    evicted_idx = atten_idxs[head_index_helper, evicted_pos]
                    free_idx = torch.stack((layer_index_helper, head_index_helper, evicted_idx), dim=-1).view(-1,3)
                
                    ## 更新缓存索引映射表和累积注意力分数
                    head_index_helper = torch.arange(selected_pos.shape[0], device='cuda').repeat(selected_pos.shape[1], 1).t()
                    req_to_atten_indexs[self.layer_num_][req_idx, :, :cache_size] = atten_idxs[head_index_helper, selected_pos] # token_idxs[selected_pos]
                    req_to_atten_scores[self.layer_num_][req_idx, :, :cache_size] = accum_score[head_index_helper, selected_pos]
                    req_to_atten_times [self.layer_num_][req_idx, :, :cache_size] = accum_times[head_index_helper, selected_pos]
                    req_to_atten_indexs[self.layer_num_][req_idx, :, cache_size:] = 0
                    req_to_atten_scores[self.layer_num_][req_idx, :, cache_size:] = 0.0
                    req_to_atten_times [self.layer_num_][req_idx, :, cache_size:] = 0

                    ## 释放掉被丢弃的token的对应缓存
                    infer_state.mem_manager.free_finegrained_by_index(free_idx)
            ###############################################################################################



    @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_attention(self, input_embding, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        cache_k, cache_v = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v  = self._get_qkv(input1, cache_k, cache_v, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))
        return

    def _context_attention_cache_dropping(self, input_embding, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        cache_k, cache_v = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v  = self._get_qkv(input1, cache_k, cache_v, infer_state, layer_weight)
        input1 = None
        att_score, o = self._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight, return_att_score=True)
        q = None
        dropped_cache_k, dropped_cache_v = self._drop_cache(att_score, cache_k, cache_v, infer_state)
        self._post_cache_kv(dropped_cache_k, dropped_cache_v, infer_state, layer_weight)  # 直接存入压缩后的kv
        att_score = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))
        return

    @mark_cost_time("trans context ffn forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_ffn(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return

    # this impl dont to use @mark_cost_time
    def _token_attention(self, input_embding, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        cache_k, cache_v = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v = self._get_qkv(input1, cache_k, cache_v, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))
        return

    def _token_attention_cache_dropping(self, input_embding, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        cache_k, cache_v = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v = self._get_qkv(input1, cache_k, cache_v, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight) # 先把当前tokens的kv值存进cache
        att_score, o = self._token_attention_kernel(q, infer_state, layer_weight, return_att_score=True) # 计算attention
        q = None
        self._drop_cache(att_score, cache_k, cache_v, infer_state)
        att_score = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))
        return


    # this impl dont to use @mark_cost_time
    def _token_ffn(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return
    
    # @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _splitfuse_attention(self, input_embding, infer_state: SplitFuseInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        cache_k, cache_v = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v  = self._get_qkv(input1, cache_k, cache_v, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o = self._splitfuse_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))
        return

    # @mark_cost_time("trans context ffn forward time cost")  # dont to remove this, will make performence down, did not know why
    def _splitfuse_ffn(self, input_embdings, infer_state: SplitFuseInferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return
    
    def context_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        if os.getenv('ENABLE_CACHE_DROPPING') == '1':
            self._context_attention = self._context_attention_cache_dropping
        self._context_attention(input_embdings,
                                      infer_state,
                                      layer_weight=layer_weight)
        self._context_ffn(input_embdings, infer_state, layer_weight)
        return input_embdings

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        if os.getenv('ENABLE_CACHE_DROPPING') == '1':
            self._token_attention = self._token_attention_cache_dropping
        self._token_attention(input_embdings,
                                    infer_state,
                                    layer_weight=layer_weight)
        self._token_ffn(input_embdings, infer_state, layer_weight)
        return input_embdings

    def splitfuse_forward(self, input_embdings, infer_state: SplitFuseInferStateInfo, layer_weight):
        self._splitfuse_attention(input_embdings,
                            infer_state,
                            layer_weight=layer_weight)
        self._splitfuse_ffn(input_embdings, infer_state, layer_weight)
        return input_embdings
