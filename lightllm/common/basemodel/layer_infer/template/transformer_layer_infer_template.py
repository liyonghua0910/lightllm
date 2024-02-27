import logging
import os
import torch
import torch.distributed as dist
from ..transformer_layer_infer import TransformerLayerInfer
from ...infer_struct import InferStateInfo
from ...splitfuse_infer_struct import SplitFuseInferStateInfo
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv
from typing import Tuple

# if os.getenv('LIGHTLLM_DEBUG') == '1':
#     import debugpy; debugpy.connect(('10.119.39.56', 5678))

from lightllm.utils.log_utils import init_logger
logger = init_logger(__name__)
logger.setLevel(logging.DEBUG if os.getenv('LIGHTLLM_DEBUG')=='1' else logging.INFO)


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
        if infer_state.mem_is_contiguous:
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
        if not infer_state.mem_is_contiguous:
            self._copy_kv_to_mem_cache(cache_k, cache_v, infer_state.mem_index, mem_manager)
            return

    def _copy_kv_to_mem_cache(self, key_buffer, value_buffer, mem_index, mem_manager):
        destindex_copy_kv(key_buffer, mem_index, mem_manager.key_buffer[self.layer_num_])
        destindex_copy_kv(value_buffer, mem_index, mem_manager.value_buffer[self.layer_num_])
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

    def _h2_eviction(self, att_score, infer_state: InferStateInfo):

        batch, head_num, _ = att_score.shape
        req_to_token_indexs = infer_state.req_manager.req_to_token_indexs
        req_to_atten_indexs = infer_state.req_manager.req_to_atten_indexs
        req_to_atten_scores = infer_state.req_manager.req_to_atten_scores

        cache_size = infer_state.req_manager.cache_size
        sink_size = infer_state.req_manager.cache_sink_size
        top_size = infer_state.req_manager.cache_top_size
        local_size = infer_state.req_manager.cache_local_size
        
        free_idxs = [] # 需要从cache释放的token的索引

        if infer_state.is_prefill:
            for i in range(batch):  # 逐个序列分别处理（效率比较低，是否可以优化成并行处理？）
                req_idx = infer_state.b_req_idx[i]
                seq_len = infer_state.b_seq_len[i]
                att_len = infer_state.b_att_len[i]
                token_idxs = req_to_token_indexs[req_idx, :seq_len]
                score = att_score[i, :, :seq_len]
                assert seq_len == att_len, 'In prefill phase, all tokens pay attention to each other.'

                if att_len <= cache_size:
                    # 如果缓存预算足够, 那么保留全部tokens的状态
                    req_to_atten_indexs[req_idx, self.layer_num_, :, :att_len] = token_idxs.repeat(head_num,1)
                    req_to_atten_scores[req_idx, self.layer_num_, :, :att_len] = score
                    free_idxs.append(torch.empty(0, device='cuda'))
                else:
                    # 如果缓存预算不够, 那么缓存由sink+top+local三部分组成
                    top_pos = sink_size + torch.argsort(score[:, sink_size:att_len-local_size], descending=True)[:,:top_size]  # 在sink+local之外选择top
                    sink_pos = torch.arange(0, sink_size, device=top_pos.device).repeat(head_num, 1)  # 生成sink的下标
                    local_pos = torch.arange(att_len-local_size, att_len, device=top_pos.device).repeat(head_num,1)  # 生成local的下标
                    selected_pos = torch.concat((sink_pos, top_pos, local_pos), dim=1)
                    selected_pos = torch.sort(selected_pos, descending=False).values
                    assert selected_pos.shape == (head_num, cache_size)
                    req_to_atten_indexs[req_idx, self.layer_num_, :, :cache_size] = token_idxs[selected_pos]
                    req_to_atten_scores[req_idx, self.layer_num_, :, :cache_size] = score[torch.arange(head_num).repeat(cache_size,1).transpose(0,1), selected_pos]
                    
                    # 确定哪些tokens占用的缓存预算需要被释放掉
                    selected_pos_bc = torch.bincount(selected_pos.flatten(), minlength=att_len)
                    free_pos = torch.where(selected_pos_bc == 0)
                    free_idx = token_idxs[free_pos]
                    free_idxs.append(free_idx)

        else:
            for i in range(batch):  # 逐个序列分别处理
                req_idx = infer_state.b_req_idx[i]
                seq_len = infer_state.b_seq_len[i]
                att_len = infer_state.b_att_len[i]
                score = att_score[i, :, :att_len]

                if att_len <= cache_size:
                    # 如果缓存预算还没满, 那就保留上一步生成的 token
                    req_to_atten_indexs[req_idx, self.layer_num_, :, att_len-1] = req_to_token_indexs[req_idx, seq_len-1]
                    req_to_atten_scores[req_idx, self.layer_num_, :, att_len-1] = 0.
                    req_to_atten_scores[req_idx, self.layer_num_, :, :att_len] += score
                    free_idxs.append(torch.empty(0, device='cuda'))
                else:
                    # 如果缓存预算已经用满, 那么用最久远的 local token 替换分数最低的 h2 token, 并且加入最新的 token 
                    assert att_len == cache_size + 1
                    req_to_atten_scores[req_idx, self.layer_num_] += score[:, :att_len-1]
                    old_atten_idx_bc = req_to_atten_indexs[req_idx, self.layer_num_].flatten().bincount()
                    sink_indexs, top_indexs, local_indexs = torch.split(req_to_atten_indexs[req_idx, self.layer_num_], [sink_size, top_size, local_size], dim=1)
                    sink_scores, top_scores, local_scores = torch.split(req_to_atten_scores[req_idx, self.layer_num_], [sink_size, top_size, local_size], dim=1)
                    # 更新top tokens, 注意力分数最低的token会被驱逐
                    if top_size > 0:
                        evicted_score, evicted_pos = torch.min(top_scores, dim=1)
                        for head in range(head_num):
                            if local_size > 0 and evicted_score[head] < local_scores[head, 0]:  
                                # 如果cache dropping策略包含local, 那么用最老的local token替换
                                top_indexs[head, evicted_pos[head]] = local_indexs[head, 0]
                                top_scores[head, evicted_pos[head]] = local_scores[head, 0]
                            elif local_size == 0 and evicted_score[head] < score[head, att_len-1]:
                                # 如果cache dropping策略不包含local, 那么用上一步生成的新token替换
                                top_indexs[head, evicted_pos[head]] = req_to_token_indexs[req_idx, seq_len-1]
                                top_scores[head, evicted_pos[head]] = score[head, att_len-1]
                    # 更新local tokens, 去除最老的token, 并加入最新的token
                    local_indexs = req_to_token_indexs[req_idx, seq_len-local_size:seq_len].repeat(head_num, 1)
                    local_scores = torch.concat((local_scores[:, 1:], score[:, att_len-int(local_size>0):att_len]), dim=1)
                    assert local_indexs.shape[-1] == local_scores.shape[-1] == local_size
                    req_to_atten_indexs[req_idx, self.layer_num_] = torch.concat((sink_indexs, top_indexs, local_indexs), dim=1)
                    req_to_atten_scores[req_idx, self.layer_num_] = torch.concat((sink_scores, top_scores, local_scores), dim=1)

                    # 确定哪些tokens占用的缓存预算需要被释放掉
                    new_atten_idx_bc = torch.bincount(req_to_atten_indexs[req_idx, self.layer_num_].flatten())
                    assert old_atten_idx_bc.sum() == new_atten_idx_bc.sum()
                    diff_len = len(new_atten_idx_bc) - len(old_atten_idx_bc)
                    if diff_len > 0:
                        old_atten_idx_bc = torch.nn.functional.pad(old_atten_idx_bc, [0, diff_len])
                    else:
                        new_atten_idx_bc = torch.nn.functional.pad(new_atten_idx_bc, [0, -diff_len])
                    free_idx = torch.where((new_atten_idx_bc == 0) & (old_atten_idx_bc != 0))[0]                  
                    free_idxs.append(free_idx)

                    cached_portion = new_atten_idx_bc.count_nonzero() / seq_len
                    # logger.debug(f'seq {i} len {seq_len.item()}, layer {self.layer_num_} cached portion: {cached_portion*100:.1f}%')

        free_idxs = torch.concat(free_idxs)
        if self.layer_num_ == 0 and self.tp_rank_ == 0:
            logger.debug(f'layer {self.layer_num_}, head 0, att_seq {req_to_atten_indexs[req_idx, self.layer_num_, 0].tolist()}')
        # logger.debug(f'b_seq_len {infer_state.b_seq_len.tolist()}, layer {self.layer_num_} delete tokens: {free_idxs.tolist()}')
        return


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

    def _context_attention_h2o(self, input_embding, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        cache_k, cache_v = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v  = self._get_qkv(input1, cache_k, cache_v, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        att_score, o = self._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight, return_att_score=True)
        q = None
        self._h2_eviction(att_score, infer_state)
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

    def _token_attention_h2o(self, input_embding, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        cache_k, cache_v = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v = self._get_qkv(input1, cache_k, cache_v, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight) # 先把当前tokens的kv值存进cache
        att_score, o = self._token_attention_kernel(q, infer_state, layer_weight, return_att_score=True) # 计算attention
        q = None
        self._h2_eviction(att_score, infer_state) # 从cache中释放不需要的tokens
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
        if os.getenv('ENABLE_HEAVY_HITTER_ORACLE') == '1':
            self._context_attention = self._context_attention_h2o
        self._context_attention(input_embdings,
                                      infer_state,
                                      layer_weight=layer_weight)
        self._context_ffn(input_embdings, infer_state, layer_weight)
        return input_embdings

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        if os.getenv('ENABLE_HEAVY_HITTER_ORACLE') == '1':
            self._token_attention = self._token_attention_h2o
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
