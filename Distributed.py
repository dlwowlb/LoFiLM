#멀티 GPU 사용시 필수

import torch
from torch import nn
from torch.autograd import Function
import torch.distributed as dist

from einops import rearrange

# distributed helpers

def all_gather_same_dim(t):
    world_size = dist.get_world_size() # 분산 환경에서 프로세스 개수(GPU 넘버링 개수)
    #t와 동일한 모양,장치,타입을 만들어 놓음
    gathered_tensors = [torch.empty_like(t, device = t.device, dtype = t.dtype) for i in range(world_size)]
    #gathered_tensors에 t를 복사 -> 이해 안됨
    dist.all_gather(gathered_tensors, t)
    return gathered_tensors

#텐서가 가변 길이를 가졌을때 사용 최대 길이에 맞춰 패딩 -> all-gather 후 -> 패딩 제거
def all_gather_variable_dim(t, dim = 0, sizes = None):
    device, rank, world_size = t.device, dist.get_rank(), dist.get_world_size() #rank는 현재 프로세스의 GPU 넘버링

    if not exists(sizes):
        size = torch.tensor(t.shape[dim], device = device, dtype = torch.long)
        sizes = all_gather_same_dim(size)
        sizes = torch.stack(sizes)

    if torch.unique(sizes).numel() == 1:
        gathered_tensors = all_gather_same_dim(t)
        return torch.cat(gathered_tensors, dim = dim), sizes

    max_size = sizes.amax().item()

    padded_t = pad_dim_to(t, max_size, dim = dim)
    gathered_tensors = all_gather_same_dim(padded_t)

    gathered_tensor = torch.cat(gathered_tensors, dim = dim)
    seq = torch.arange(max_size, device = device)

    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    mask = rearrange(mask, 'i j -> (i j)')
    seq = torch.arange(mask.shape[-1], device = device)
    indices = seq[mask]

    gathered_tensor = gathered_tensor.index_select(dim, indices)

    return gathered_tensor, sizes


# forward에서 여러 rank 텐서를 모으고, backward에서 그래디언트를 적절히 분배(split)하는 역할
class AllGatherFunction(Function):
    @staticmethod
    def forward(ctx, x, dim, sizes, all_reduce_grads):
        x, batch_sizes = all_gather_variable_dim(x, dim = dim, sizes = sizes)
        ctx.dim = dim
        ctx.all_reduce_grads = all_reduce_grads
        ctx.batch_sizes = batch_sizes.tolist()
        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        if ctx.all_reduce_grads:
            dist.all_reduce(grads)

        grads_by_rank = grads.split(batch_sizes, dim = ctx.dim)
        return grads_by_rank[rank], None, None, None

class AllGather(nn.Module):
    def __init__(
        self,
        dim,
        *,
        all_reduce_grads = False
    ):
        super().__init__()
        self.dim = dim
        self.all_reduce_grads = all_reduce_grads
        #싱글 프로세스라면 이 모듈은 의미가 없으므로, 별도 동작 안 함.
        self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

    def forward(
        self,
        x,
        sizes = None
    ):
        if not self.is_distributed:
            return x, None

        return AllGatherFunction.apply(x, self.dim, sizes, self.all_reduce_grads)