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
        
        self.min_cache_size = int(os.getenv('MIN_CACHE_SIZE'))  # 最小的缓存长度，限制序列较长且压缩率低的情况
        self.max_cache_size = int(os.getenv('MAX_CACHE_SIZE'))  # 最大的缓存长度，限制序列较长且压缩率高的情况
        self.compression_rate = float(os.getenv('COMPRESSION_RATE'))  # 缓存的压缩率 = 压缩后的缓存大小 / 序列的prompt长度

        # 为kv cache压缩维护的请求状态
        self.req_to_atten_indexs = [torch.zeros((max_request_num, att_head_num, self.max_cache_size + 1), dtype=torch.int32, device="cuda") for _ in range(layers_num)]
        self.req_to_cum_scores = [torch.zeros((max_request_num, att_head_num, self.max_cache_size + 1), dtype=torch.float16, device="cuda") for _ in range(layers_num)]
        self.req_to_cum_times = [torch.zeros((max_request_num, att_head_num, self.max_cache_size + 1), dtype=torch.int32, device="cuda") for _ in range(layers_num)]
        self.req_to_cache_usage = [torch.zeros((max_request_num,), dtype=torch.int32, device='cuda') for _ in range(layers_num)]

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
            self.req_to_atten_indexs[layer][free_req_index, :, :] = -1
            self.req_to_cum_scores[layer][free_req_index, :, :] = 0
            self.req_to_cum_times[layer][free_req_index, :, :] = 0
            self.req_to_cache_usage[layer][free_req_index] = 0
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
    
    