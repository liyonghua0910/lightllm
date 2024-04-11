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

if os.getenv('ENABLE_DEBUGPY') == '1':
    import debugpy; debugpy.connect(('10.119.25.81', 5678))

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

    def _get_cache_recipe(self, b_cache_usage, recipe=None):
        recipe = os.getenv('CACHE_RECIPE')
        if recipe == 'H2O':
            b_sink_usage = torch.zeros_like(b_cache_usage)
            b_top_usage = b_cache_usage // 2
            b_local_usage = b_cache_usage - b_top_usage
        elif recipe == 'StreamingLLM':
            b_sink_usage = torch.full_like(b_cache_usage, 4)
            b_top_usage = torch.zeros_like(b_cache_usage)
            b_local_usage = b_cache_usage - b_sink_usage
        elif recipe == 'Hybrid':
            b_sink_usage = torch.full_like(b_cache_usage, 4)
            b_top_usage = (b_cache_usage - b_sink_usage) // 2
            b_local_usage = b_cache_usage - b_sink_usage - b_top_usage
        else:
            raise NotImplementedError('Unsupported cache recipe! Expected one of [H2O, StreamingLLM, Hybrid]')
        return b_sink_usage, b_top_usage, b_local_usage

    def _get_score_for_sort(self, cur_score, cum_score, atten_times):
        sort_by = os.getenv('ATTENTION_SORT_BY')
        if sort_by == 'average':
            return cum_score / atten_times
        elif sort_by == 'cummulative':
            return cum_score
        elif sort_by == 'volatile':
            return cur_score
        else:
            raise NotImplementedError('Unsupported sorting policy! Expected one of [average, cummulative, volatile]')

    def _drop_cache(self, cur_score, cum_score, cache_k, cache_v, infer_state: InferStateInfo):

        layer = self.layer_num_
        batch_size, head_num, _ = cur_score.shape
        req_to_atten_indexs = infer_state.req_manager.req_to_atten_indexs
        req_to_cum_scores = infer_state.req_manager.req_to_cum_scores
        req_to_cum_times = infer_state.req_manager.req_to_cum_times
        req_to_cache_usage = infer_state.req_manager.req_to_cache_usage
        
        b_att_len = infer_state.b_att_len
        b_req_idx = infer_state.b_req_idx.long()
        b_start_loc = infer_state.b_start_loc
        b_cache_usage = req_to_cache_usage[layer][b_req_idx]

        if infer_state.is_prefill:
            cached_kv_pos = [None for _ in range(batch_size)]
            
            ## 批量处理不需要丢弃词的序列
            no_evict_req_ids = torch.nonzero(b_att_len[layer] <= b_cache_usage).squeeze(-1)
            no_evict_req_num = no_evict_req_ids.numel()
            if no_evict_req_num > 0:
                no_evict_att_len = b_att_len[layer][no_evict_req_ids]
                no_evict_max_att_len = no_evict_att_len.max().item()
                no_evict_cum_score = cum_score[no_evict_req_ids, :, :no_evict_max_att_len]
                no_evict_cum_times = torch.zeros_like(no_evict_cum_score, dtype=torch.int32)
                for i, j in enumerate(no_evict_att_len):
                    no_evict_cum_times[i, :, :j] = torch.arange(j, 0, -1).cuda()
                # 填充各token的累积注意力分数、各token被关注的次数
                no_evict_req_idxs = b_req_idx[no_evict_req_ids]
                req_to_cum_scores[layer][no_evict_req_idxs, :, :no_evict_max_att_len] = no_evict_cum_score
                req_to_cum_times[layer][no_evict_req_idxs, :, :no_evict_max_att_len] = no_evict_cum_times
                # 收集需要缓存的kv在buffer中的位置
                no_evict_start_loc = b_start_loc[no_evict_req_ids]
                for i, j in enumerate(no_evict_req_ids):
                    cached_kv_pos[j] = no_evict_start_loc[i] + torch.arange(0, no_evict_att_len[i].item()).repeat(head_num, 1).cuda()
            
            ## 批量处理需要丢弃词的序列
            to_evict_req_ids = torch.nonzero(b_att_len[layer] > b_cache_usage).squeeze(-1)
            to_evict_req_num = to_evict_req_ids.numel()
            if to_evict_req_num > 0:
                to_evict_req_idxs = b_req_idx[to_evict_req_ids]
                to_evict_att_len = b_att_len[layer][to_evict_req_ids]
                to_evict_max_att_len = to_evict_att_len.max().item()
                to_evict_cur_score = cur_score[to_evict_req_ids, :, :to_evict_max_att_len]
                to_evict_cum_score = cum_score[to_evict_req_ids, :, :to_evict_max_att_len]
                to_evict_cum_times = torch.zeros_like(to_evict_cum_score, dtype=torch.int32)
                for i, j in enumerate(to_evict_att_len):
                    to_evict_cum_times[i, :, :j] = torch.arange(j, 0, -1).cuda()
                # 计算cache中各组成成分的token数量
                to_evict_cache_usage = b_cache_usage[to_evict_req_ids]
                to_evict_sink_usage, _, to_evict_local_usage = self._get_cache_recipe(to_evict_cache_usage)
                # 构建topk候选词的掩码: 在序列有效长度以内、非sink、非local
                candidate_mask = torch.zeros_like(to_evict_cum_score)
                for i in range(to_evict_req_num):
                    candidate_mask[i, :, :to_evict_att_len[i]] = 1
                    candidate_mask[i, :, :to_evict_sink_usage[i]] = -1
                    candidate_mask[i, :, to_evict_att_len[i]-to_evict_local_usage[i]:to_evict_att_len[i]] = -1
                # 排序选出所需要的token在当前序列中的下标
                to_evict_score_for_sort = self._get_score_for_sort(to_evict_cur_score, to_evict_cum_score, to_evict_cum_times)
                to_evict_score_for_sort = torch.where(candidate_mask == 1, to_evict_score_for_sort, -torch.inf)
                to_evict_score_for_sort = torch.where(candidate_mask == -1, torch.inf, to_evict_score_for_sort)
                to_evict_sorted_pos = torch.argsort(to_evict_score_for_sort, descending=True, stable=True)
                # 填充各token的累积注意力分数、填充各token被关注的次数
                for i in range(to_evict_req_num):
                    cache_usage = to_evict_cache_usage[i]
                    selected_pos = to_evict_sorted_pos[i][:,:cache_usage].sort().values
                    idx_h = torch.arange(head_num).unsqueeze(-1).expand_as(selected_pos).cuda()
                    req_to_cum_scores[layer][to_evict_req_idxs[i], :, :cache_usage] = to_evict_cum_score[i][idx_h, selected_pos]
                    req_to_cum_times[layer][to_evict_req_idxs[i], :, :cache_usage] = to_evict_cum_times[i][idx_h, selected_pos]
                    cached_kv_pos[to_evict_req_ids[i]] = b_start_loc[to_evict_req_ids[i]] + selected_pos
            
            ## 取出真正需要缓存的kv
            cached_kv_pos = torch.concat(cached_kv_pos, dim=-1)
            dropped_cache_k = cache_k[cached_kv_pos, torch.arange(self.tp_k_head_num_).repeat(cached_kv_pos.shape[1], 1).t()].transpose(0,1).contiguous()
            dropped_cache_v = cache_v[cached_kv_pos, torch.arange(self.tp_v_head_num_).repeat(cached_kv_pos.shape[1], 1).t()].transpose(0,1).contiguous()
            return dropped_cache_k, dropped_cache_v

        else:
            assert cum_score == None
            # 批量处理不需要丢弃词的序列
            no_evict_req_ids = torch.nonzero(b_att_len[layer] <= b_cache_usage).squeeze(-1)
            no_evict_req_num = no_evict_req_ids.numel()
            if no_evict_req_num > 0:
                no_evict_req_idxs = b_req_idx[no_evict_req_ids]
                no_evict_att_len = b_att_len[layer][no_evict_req_ids]
                no_evict_max_att_len = no_evict_att_len.max().item()
                no_evict_cum_score = cur_score[no_evict_req_ids, :, :no_evict_max_att_len].clone()
                no_evict_cum_score[:,:,:-1] += req_to_cum_scores[layer][no_evict_req_idxs, :, :no_evict_max_att_len-1]
                no_evict_cum_times = torch.ones_like(no_evict_cum_score, dtype=torch.int32)
                no_evict_cum_times[:,:,:-1] += req_to_cum_times[layer][no_evict_req_idxs, :, :no_evict_max_att_len-1]
                # 填充各token的累积注意力分数、填充各token被关注的次数
                req_to_cum_scores[layer][no_evict_req_idxs, :, :no_evict_max_att_len] = no_evict_cum_score
                req_to_cum_times[layer][no_evict_req_idxs, :, :no_evict_max_att_len] = no_evict_cum_times

            # 批量处理需要丢弃词的序列
            to_evict_req_ids = torch.nonzero(b_att_len[layer] > b_cache_usage).squeeze(-1)
            to_evict_req_num = to_evict_req_ids.numel()
            if to_evict_req_num > 0:
                to_evict_req_idxs = b_req_idx[to_evict_req_ids]
                to_evict_att_len = b_att_len[layer][to_evict_req_ids]
                to_evict_max_att_len = to_evict_att_len.max().item()
                to_evict_atten_indexs = req_to_atten_indexs[layer][to_evict_req_idxs, :, :to_evict_max_att_len]
                to_evict_cur_score = cur_score[to_evict_req_ids, :, :to_evict_max_att_len].clone()
                to_evict_cum_score = cur_score[to_evict_req_ids, :, :to_evict_max_att_len].clone()
                to_evict_cum_score[:,:,:-1] += req_to_cum_scores[layer][to_evict_req_idxs, :, :to_evict_max_att_len-1]
                to_evict_cum_times = torch.ones_like(to_evict_cum_score, dtype=torch.int32)
                to_evict_cum_times[:,:,:-1] += req_to_cum_times[layer][to_evict_req_idxs, :, :to_evict_max_att_len-1]
                # 计算cache中各组成成分的token数量
                to_evict_cache_usage = b_cache_usage[to_evict_req_ids]
                to_evict_sink_usage, _, to_evict_local_usage = self._get_cache_recipe(to_evict_cache_usage)
                assert (to_evict_att_len == to_evict_cache_usage + 1).all(), "在decode阶段, 缓存已满的注意力序列的长度应该恰好是缓存序列的长度加一"
                # 构建topk候选词的掩码: 在序列有效长度以内、非sink、非local
                candidate_mask = torch.zeros_like(to_evict_cum_score)
                for i in range(to_evict_req_num):
                    candidate_mask[i, :, :to_evict_att_len[i]] = 1
                    candidate_mask[i, :, :to_evict_sink_usage[i]] = -1
                    candidate_mask[i, :, to_evict_att_len[i]-to_evict_local_usage[i]:to_evict_att_len[i]] = -1
                assert (candidate_mask.count_nonzero(dim=-1) == to_evict_att_len.unsqueeze(dim=-1)).all(), "注意力序列的所有token都是候选者"
                # 排序选出所需要的token在当前序列中的下标
                to_evict_score_for_sort = self._get_score_for_sort(to_evict_cur_score, to_evict_cum_score, to_evict_cum_times)
                to_evict_score_for_sort = torch.where(candidate_mask == 1, to_evict_score_for_sort, -torch.inf)
                to_evict_score_for_sort = torch.where(candidate_mask == -1, torch.inf, to_evict_score_for_sort)
                to_evict_sorted_pos = torch.argsort(to_evict_score_for_sort, descending=True, stable=True)
                # 在缓存索引映射表更新之前，确定需要被释放的token的索引，并释放对应的缓存
                # 然后更新缓存索引映射表、更新各token的累积注意力分数、更新各token被关注的次数
                for i in range(to_evict_req_num):
                    att_len = to_evict_att_len[i]
                    cache_usage = to_evict_cache_usage[i]
                    selected_pos = to_evict_sorted_pos[i, :, :cache_usage].sort().values
                    evicted_pos = to_evict_sorted_pos[i, :, cache_usage:att_len]
                    assert evicted_pos.size(-1) == 1, "在decode阶段, 缓存已满的序列的只需要驱逐一个token"
                    idx_h = torch.arange(head_num).unsqueeze(-1).expand_as(evicted_pos).cuda()
                    evicted_idx = to_evict_atten_indexs[i, idx_h, evicted_pos]
                    free_idx = torch.stack([torch.full_like(evicted_pos, layer), idx_h, evicted_idx]).view(3,-1).transpose(0,1)
                    assert (infer_state.mem_manager.finegrained_mem_state[free_idx.long().t().tolist()] == 1).all(), "必须已经被占用的缓存位置才能被释放"
                    infer_state.mem_manager.free_finegrained_by_index(free_idx)
                    idx_h = torch.arange(head_num).unsqueeze(-1).expand_as(selected_pos).cuda()
                    req_to_atten_indexs[layer][to_evict_req_idxs[i], :, :cache_usage] = to_evict_atten_indexs[i, idx_h, selected_pos]
                    req_to_cum_scores[layer][to_evict_req_idxs[i], :, :cache_usage] = to_evict_cum_score[i, idx_h, selected_pos]
                    req_to_cum_times[layer][to_evict_req_idxs[i], :, :cache_usage] = to_evict_cum_times[i, idx_h, selected_pos]
            
            return None, None

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
        cur_score, cum_score, o = self._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight, return_att_score=True)
        q = None
        dropped_cache_k, dropped_cache_v = self._drop_cache(cur_score, cum_score, cache_k, cache_v, infer_state)
        self._post_cache_kv(dropped_cache_k, dropped_cache_v, infer_state, layer_weight)  # 直接存入压缩后的kv
        cur_score = None
        cum_score = None
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
        cur_score, cum_score, o = self._token_attention_kernel(q, infer_state, layer_weight, return_att_score=True) # 计算attention
        q = None
        self._drop_cache(cur_score, cum_score, cache_k, cache_v, infer_state)
        cur_score = None
        cum_score = None
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
