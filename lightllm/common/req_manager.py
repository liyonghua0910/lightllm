import torch
import os
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

class ReqManager:
    def __init__(self, max_request_num, max_sequence_length, layers_num, att_head_num, att_head_dim, mem_manager):
        self.req_state = torch.zeros((max_request_num,), dtype=torch.bool, device="cuda")
        self.req_to_token_indexs = torch.zeros((max_request_num, max_sequence_length), dtype=torch.int32, device="cuda")
        self.can_use_req_size = max_request_num
        self.mem_manager = mem_manager
        self.att_head_num = att_head_num
        self.layers_num = layers_num
        
        self.cache_sink_size = int(os.getenv("CACHE_SINK_SIZE"))
        self.cache_top_size = int(os.getenv("CACHE_TOP_SIZE"))
        self.cache_local_size = int(os.getenv("CACHE_LOCAL_SIZE"))
        self.cache_size = self.cache_sink_size + self.cache_top_size + self.cache_local_size

        # 分层压缩, 前2层不压缩, 后面层压缩为sink+top+local
        self.layers_cache_size = [max_sequence_length] * 2 + [self.cache_size] * (layers_num - 2)
        self.layers_cache_sink_size = [0] * 2 + [self.cache_sink_size] * (layers_num - 2)
        self.layers_cache_top_size = [0] * 2 + [self.cache_top_size] * (layers_num - 2)
        self.layers_cache_local_size = [max_sequence_length] * 2 + [self.cache_local_size] * (layers_num - 2)
        
        # 为kv cache压缩维护的请求状态
        self.req_to_atten_indexs = [torch.zeros((max_request_num, att_head_num, layer_cache_size + 1), dtype=torch.int32, device="cuda") for layer_cache_size in self.layers_cache_size]
        self.req_to_atten_scores = [torch.zeros((max_request_num, att_head_num, layer_cache_size + 1), dtype=torch.float16, device="cuda") for layer_cache_size in self.layers_cache_size]
        self.req_to_atten_times = [torch.zeros((max_request_num, att_head_num, layer_cache_size + 1), dtype=torch.int32, device="cuda") for layer_cache_size in self.layers_cache_size]

    def alloc(self, need_size):
        if need_size > self.can_use_req_size:
            logger.error(f'Insufficient requested capacity, remaining {self.can_use_req_size}')
            return None
        select_index = torch.nonzero(self.req_state==0).reshape(-1)[:need_size]
        self.req_state[select_index] = 1
        self.can_use_req_size -= len(select_index)
        return select_index
    
    def free(self, free_req_index, free_token_index, free_atten_index=None):
        self.can_use_req_size += len(free_req_index)
        self.req_state[free_req_index] = 0
        for layer in range(self.layers_num):
            self.req_to_atten_indexs[layer][free_req_index, :, :] = 0
            self.req_to_atten_scores[layer][free_req_index, :, :] = 0
            self.req_to_atten_times[layer][free_req_index, :, :] = 0
        if self.can_use_req_size == len(self.req_state):
            logger.debug(f"freed all request size {self.can_use_req_size}")
        self.mem_manager.free(free_token_index)
        self.mem_manager.free_finegrained_by_index(free_atten_index)
    
    def free_req(self, free_req_index):
        self.can_use_req_size +=1
        self.req_state[free_req_index] = 0
        return
    
    def free_token(self, free_token_index):
        self.mem_manager.free(free_token_index)

    def free_all(self):
        self.can_use_req_size = len(self.req_state)
        self.req_state[:] = 0
    
    