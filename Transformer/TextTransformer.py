from beartype.typing import List, Optional, Tuple
from beartype import beartype
from x_clip.tokenizer import tokenizer

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops.layers.torch import Rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

class LayerNorm(nn.Module):
    def __init__(self, dim, scale = True):
        super().__init__()
        self.learned_gamma = nn.Parameter(torch.ones(dim)) if scale else None

        self.register_buffer('gamma', torch.ones(dim), persistent = False)
        self.register_buffer('beta', torch.zeros(dim), persistent = False)

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], default(self.learned_gamma, self.gamma), self.beta)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

#GEGLU를 이용한 FeedForward
def FeedForward(dim, mult = 4, dropout = 0.):
    dim_hidden = int(dim * mult * 2 / 3)

    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, dim_hidden * 2, bias = False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim, bias = False)
    )


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        scale = 8
    ):
        super().__init__()
        self.heads = heads
        self.scale = scale
        self.causal = causal
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None
    ):
        b, n, _, device = *x.shape, x.device

        # prenorm
        x = self.norm(x)

        # project for queries, keys, values
        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim = -1)

        # split for multi-headed attention
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        # qk rmsnorm, technique circulating within brain used to stabilize a 22B parameter vision model training

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # similarities(attention scores)
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention
        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout),
            ]))

    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None,
        return_all_layers = False
    ):
        layers = []

        # residual 연결을 사용하여 각 layer를 연결
        # depth만큼 (attention과 feedforward)를 반복
        for attn, ff in self.layers:
            x = attn(x, rel_pos_bias = rel_pos_bias, mask = mask) + x
            x = ff(x) + x
            layers.append(x)

        if not return_all_layers:
            return x

        return x, torch.stack(layers[:-1])



class textTransformer:
    @beartype
    def __init__(
        self,
        dim,
        depth,
        num_tokens = tokenizer.vocab_size,
        max_seq_len = 256,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        pad_id = 0
        ):

        super().__init__()
        self.dim = dim
        
        #토큰 임베딩과 포지션 임베딩
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.cls_token = nn.Parameter(torch.randn(dim))

        self.pad_id = pad_id

    @beartype
    def forward(
            self,
            x = None,
            raw_texts: Optional[List[str]] = None,
            mask = None,
            return_all_layers = False
    ):
        assert exists(x) ^ exists(raw_texts) # raw text 혹은 x가 존재해야 함

        if exists(raw_texts):
            #openAI tokenizer를 통해 raw text를 tokenize
            x = tokenizer.tokenize(raw_texts).to(self.device)

        #패딩 토큰(pad_id)이 아니면 True, 패딩 토큰이면 False 불리언 마스크 생성
        if not exists(mask):
            mask = x != self.pad_id

        b, n, device = *x.shape, x.device
        
        #토큰 임베딩
        x = self.token_emb(x)

        #텍스트 시퀸스가 max_seq_len보다 길면 에러 발생
        assert n <= self.max_seq_len, f'text sequence length {n} must be less than {self.max_seq_len}'

        #포지션 임베딩
        x = x + self.pos_emb(torch.arange(n, device = device))

        #CLS 토큰 추가
        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)
        x, ps = pack([cls_tokens, x], 'b * d')

        #CLS 토큰 처리했으므로 마스크에도 추가
        mask = F.pad(mask, (1, 0), value = True)
        
        #Transformer 모델에 통과
        x, all_layers = self.transformer(x, mask = mask, return_all_layers = True)

        #CLS 토큰만 추출
        cls_tokens, _ = unpack(x, ps, 'b * d')
        out = self.norm(cls_tokens)
        
        if not return_all_layers:
            return out

        return out, all_layers
