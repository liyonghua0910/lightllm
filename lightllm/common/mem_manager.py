import torch
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
    

class MemoryManager:
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False):
        self.size = size        
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.always_copy = always_copy
        
        # mem_state 修改为使用计数方式，方便后期实现token共享机制，实现beam search 等
        self.mem_state = torch.zeros((size,), dtype=torch.int32, device="cuda")
        self.finegrained_mem_state = torch.zeros((layer_num, head_num, size), dtype=torch.int8, device="cuda")  # 0/1
        self.indexes = torch.arange(0, size, dtype=torch.long, device="cuda")
        self.can_use_mem_size = size
        self._init_buffers(size, dtype, head_num, head_dim, layer_num)
    
    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.key_buffer = [torch.empty((size, head_num, head_dim), dtype=dtype, device="cuda") for _ in range(layer_num)]
        self.value_buffer = [torch.empty((size, head_num, head_dim), dtype=dtype, device="cuda") for _ in range(layer_num)]
    
    def _free_buffers(self):
        self.key_buffer = None
        self.value_buffer = None

    @torch.no_grad()
    def alloc_finegrained(self, need_size):
        """ Allocate need_size memory from cache in a fine-grained manner.
            Memory state is indexed by (layer_idx, head_idx, token_idx).
            Returns a tensor shaped (layer_num, head_num, need_size).
        """
        if (need_size > (self.finegrained_mem_state == 0).sum(-1)).any():
            logger.warn(f'warn no enough cache size ({need_size}) in head!')
            return None
        alloc_idx = self.finegrained_mem_state.sort(dim=-1, descending=False, stable=True).indices[:,:,:need_size]
        layer_idx = torch.arange(self.layer_num).repeat(self.head_num * need_size, 1).t().flatten().cuda()
        head_idx = torch.arange(self.head_num).repeat(need_size, 1).t().flatten().repeat(self.layer_num).cuda()
        self.finegrained_mem_state[layer_idx, head_idx, alloc_idx.flatten()] += 1
        return alloc_idx

    @torch.no_grad()
    def free_finegrained_by_index(self, free_index_list):
        """ Free memory from cache in a fine-grained manner.
            Memory needed to be freed is specified by `free_index_list` tensor shaped (?, 3) 
            with each row looking like [layer_idx, head_idx, token_idx]
        """
        free_index = free_index_list.long().t().tolist()
        self.finegrained_mem_state[free_index] -= 1
        return

    @torch.no_grad()
    def free_finegrained(self, free_index):
        """ Free memory from cache in a fine-grained manner.
            Memory needed to be freed is specified by `free_index` tensor shaped (layer_num, head_num, ?) 
            with the last dim specifying token indices to be freed in each head
        """
        if free_index == None:
            return
        layer_num, head_num, free_size = free_index.shape
        assert layer_num == self.layer_num and head_num == self.head_num
        layer_idx = torch.arange(layer_num).repeat(head_num * free_size, 1).t().flatten().cuda()
        head_idx = torch.arange(head_num).repeat(free_size, 1).t().flatten().repeat(self.layer_num).cuda()
        self.finegrained_mem_state[layer_idx, head_idx, free_index.long().flatten()] -= 1
        return
    
    @torch.no_grad()
    def get_memory_usage(self, layer_id, head_id):
        used_mem = (self.finegrained_mem_state[layer_id, head_id] == 1).sum(-1).item()
        remained_mem = (self.finegrained_mem_state[layer_id, head_id] == 0).sum(-1).item()
        abnormal_mem_index = torch.stack(torch.where((self.finegrained_mem_state!=0)&(self.finegrained_mem_state!=1)), -1).cpu().numpy().tolist()
        return used_mem, remained_mem, abnormal_mem_index

    @torch.no_grad()
    def alloc(self, need_size):
        if need_size > self.can_use_mem_size:
            logger.warn(f'warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}')
            return None
        can_use_index = torch.nonzero(self.mem_state == 0).view(-1)
        select_index = can_use_index[0 : need_size]
        self.add_refs(select_index)
        return select_index
    
    @torch.no_grad()
    def alloc_contiguous(self, need_size):
        if self.always_copy:
            return None
        if need_size > self.can_use_mem_size:
            logger.warn(f'warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}')
            return None
        
        can_use_index = torch.nonzero(self.mem_state == 0).view(-1)
        can_use_index_size = len(can_use_index)
        can_use_index = can_use_index[0 : can_use_index_size - need_size + 1][(can_use_index[need_size - 1: ] - can_use_index[0 : can_use_index_size - need_size + 1]) == need_size - 1]
        if can_use_index.shape[0] == 0:
            # logger.warn(f'warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}')
            return None
        start = can_use_index[0].item()
        end = start + need_size
        select_index = self.indexes[start : end]
        self.add_refs(select_index)
        return select_index, start, end
    
    @torch.no_grad()
    def free(self, free_index):
        """_summary_

        Args:
            free_index (torch.Tensor): _description_
        """
        free_index = free_index.long()
        self.decrease_refs(free_index)
        if self.can_use_mem_size == len(self.mem_state):
            logger.debug(f"freed all gpu mem size {self.can_use_mem_size}")
        return
    
    @torch.no_grad()
    def add_refs(self, token_index: torch.Tensor):
        state = self.mem_state[token_index]
        has_used_tokens = torch.count_nonzero(state).item()
        all_tokens = len(state)
        self.can_use_mem_size -= all_tokens - has_used_tokens
        self.mem_state[token_index] += 1
        return
    
    @torch.no_grad()
    def decrease_refs(self, token_index: torch.Tensor):
        self.mem_state[token_index] -= 1
        state = self.mem_state[token_index]
        used_tokens = torch.count_nonzero(state).item()
        all_tokens = len(state)
        self.can_use_mem_size += all_tokens - used_tokens
        return

    
    @torch.no_grad()
    def free_all(self):
        self.can_use_mem_size = len(self.mem_state)
        self.mem_state[:] = 0
        self.finegrained_mem_state[:] = 0
    
    @torch.no_grad()
    def resize_mem(self, new_size):
        """
        just for test code
        """
        size = new_size        
        dtype = self.dtype
        head_num = self.head_num
        head_dim = self.head_dim
        layer_num = self.layer_num
        always_copy = self.always_copy

        self.mem_state = torch.zeros((size,), dtype=torch.int32, device="cuda")
        self.indexes = torch.arange(0, size, dtype=torch.long, device="cuda")
        self.can_use_mem_size = size
        self._free_buffers()
        self._init_buffers(size, dtype, head_num, head_dim, layer_num)
        return


if __name__ == "__main__":
    # Test fine-grained memory management
    mem_manager = MemoryManager(10, torch.float16, 4, 8, 2)
    print(mem_manager.finegrained_mem_state)
    mem_index = mem_manager.alloc_finegrained(3)
    print(mem_manager.finegrained_mem_state)
    mem_index = mem_manager.alloc_finegrained(10)
    print(mem_manager.finegrained_mem_state)
    free_index = torch.tensor([[0,0,0], [0,0,1], [0,0,2], [0,1,2], [0,1,0], [0,2,1]]).cuda()
    mem_index = mem_manager.free_finegrained_by_index(free_index)
    print(mem_manager.finegrained_mem_state)
    mem_index = mem_manager.alloc_finegrained(3)
