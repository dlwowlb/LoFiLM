from einops import rearrange, repeat, reduce
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchaudio.transforms import Spectrogram, TimeStretch, FrequencyMasking, TimeMasking

#patch size가 16이면 16x16 크기의 patch로
def pair(t):
    return (t, t) if not isinstance(t, tuple) else t

#n을 divisor로 나눈 후 가장 가까운 정수로 내림 
def round_down_nearest_multiple(n, divisor):
    return n // divisor * divisor


def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    #*patches.shape는 patches의 shape을 unpacking
    #h, w, dim, device, dtype에 각각 unpacking
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    #dim이 4의 배수여야 함(x에 대해 sin, cos, y에 대해 sin, cos)
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'

    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    pe = pe.type(dtype)

    return rearrange(pe, '(h w) d -> h w d', h = h, w = w)

# biasless layernorm

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

class Audiotransformer(nn.module):
    def __init__(
            self,
            dim,
            patch_size = 16,
            accept_spec = False,
            accept_spec_time_first = True,
            spec_n_fft = 128,
            spec_power = 2,
            spec_win_length = 24,
            spec_hop_length = None,
            spec_pad = 0,
            spec_center = True,
            spec_pad_mode = 'reflect',
            patch_dropout_prob = 0.25,
            spec_aug_stretch_factor = 0.8,
            spec_aug_freq_mask = 80,
            spec_aug_time_mask = 80,
            depth,
            dim_head = 64,
            heads = 8,
            attn_dropout = 0.,
            ff_mult = 4,
            ff_dropout = 0.
        ):
        super().__init__()
        self.patch_size = pair(patch_size)
        self.dim = dim

        #spectogram 라이브러리 사용
        self.spec = Spectrogram(
            n_fft = spec_n_fft,
            power = spec_power,
            win_length = spec_win_length,
            hop_length = spec_hop_length,
            pad = spec_pad,
            center = spec_center,
            pad_mode = spec_pad_mode
        )
        
        patch_input_dim = self.patch_size[0] * self.patch_size[1]


        #이미지가 32 x 32이고 patch size가 16이면 2x2 크기의 patch로 변환 후 
        # 각 patch 는 flatten되고 linear layer 적용
        self.to_patch_tokens = Sequential(
            Rearrange('b (h p1) (w p2) -> b h w (p1 p2)', p1 = self.patch_size[0], p2 = self.patch_size[1]),
            nn.LayerNorm(patch_input_dim),
            nn.Linear(patch_input_dim, dim),
            nn.LayerNorm(dim)
        )

        #다양한 augmentation을 위한 라이브러리 사용
        #TimeStretch는 스펙토그램 시간 축 늘리거나 줄임
        #FrequencyMasking은 스펙토그램 주파수 축 마스킹
        #TimeMasking은 스펙토그램 시간 축 마스킹
        self.aug = torch.nn.Sequential(
            TimeStretch(spec_aug_stretch_factor, fixed_rate = True),
            FrequencyMasking(freq_mask_param = spec_aug_freq_mask),
            TimeMasking(time_mask_param = spec_aug_time_mask),
        )

        self.accept_spec = accept_spec
        self.accept_spec_time_first = accept_spec_time_first

        #동적 위치 편향을 위한 모듈 정의
        mlp_hidden_dim = dim // 4
        self.dynamic_pos_bias_mlp = nn.Sequential(
            nn.Linear(2, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, heads),
            Rearrange('... i j h -> ... h i j')
        )

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_mult = ff_mult,
            ff_dropout = ff_dropout
        )
        

    def forward(self, x, force_no_patch_dropout = False, return_all_layers = False):
        batch, device = x.shape[0], x.device

        #x가 2차원이면 Raw waveform, 3차원이면 이미 Spectrogram

        #x의 차원이 3차원이면 accept_spec이 True여야 함
        #x의 차원이 2차원이면 accept_spec이 False여야 함
        assert (self.accept_spec and x.ndim == 3) or (not self.accept_spec and x.ndim == 2)

        #accpet_spec이 True이면 x의 차원을 변환
        if self.accept_spec and self.accept_spec_time_first:
            x = rearrange(x, 'b t f -> b f t')
        #accpet_spec이 False이면 spec을 계산
        if not self.accept_spec:
            x = self.spec(x)

        if self.accept_spec and self.accept_spec_time_first:
            x = rearrange(x, 'b f t -> b t f')
        
        if self.training:
            x = self.aug(x)

        self.patch_dropout_prob = patch_dropout_prob
        
        
        

        #Patch 연산 하도록 자동으로 크롭 
        height, width = x.shape[-2]
        patch_hegith, patch_width = self.patch_size

        #height, width를 patch size로 나눈 후 가장 가까운 정수로 내림
        rounded_height, rounded_width = map(lambda args: round_down_nearest_multiple(*args), ((height, patch_height), (width, patch_width)))

        if (height, width) != (rounded_height, rounded_width): # just keep printing to be annoying until it is fixed
            print_once(f'spectrogram yielded shape of {(height, width)}, but had to be cropped to {(rounded_height, rounded_width)} to be patchified for transformer')

        #패치 사용
        x = x[..., :rounded_height, :rounded_width]

        # 패치 토큰으로 변환
        x = self.to_patch_tokens(x)

        #패치 개수 계산
        _, num_patch_height, num_patch_width, _ = x.shape

        #패치의 위치 정보를 포함한 grid 생성
        grid = torch.stack(torch.meshgrid(
            torch.arange(num_patch_height, device = device),
            torch.arange(num_patch_width, device = device)
        , indexing = 'ij'), dim = -1)

        #각 패치 좌표를 한줄로 나열된 벡터로 변환
        grid = rearrange(grid, '... c -> (...) c')

        #위치 정보 추가
        x = x + posemb_sincos_2d(x)

        x = rearrange(x, 'b ... c -> b (...) c')
        

        #patch dropout
        #force_no_patch_dropout이면 patch dropout을 하지 않음
        if self.training and self.patch_dropout_prob > 0. and not force_no_patch_dropout:
            #n은 패치 개수
            n, device = x.shape[1], x.device


            batch_indices = torch.arange(batch, device = device)
            batch_indices = rearrange(batch_indices, '... -> ... 1') # 1차원 추가

            #유지할 패치 인덱스 선택
            num_patches_keep = max(1, int(n * (1 - self.patch_dropout_prob)))
            patch_indices_keep = torch.randn(batch, n, device = device).topk(num_patches_keep, dim = -1).indices

            x = x[batch_indices, patch_indices_keep]

            grid = repeat(grid, '... -> b ...', b = batch)
            grid = grid[batch_indices, patch_indices_keep]
        
        #patch 위치 정보를 이용한 relative position bias 계산
        rel_dist = rearrange(grid, '... i c -> ... i 1 c') - rearrange(grid, '... j c -> ... 1 j c')
        rel_pos_bias = self.dynamic_pos_bias_mlp(rel_dist.float())

        #Attention 계산산
        x, all_layers = self.transformer(x, rel_pos_bias = rel_pos_bias, return_all_layers = True)

        #글로벌 평균 풀링 (ViT에서 CLS 토큰 사용 없이 사용)
        x = reduce(x, 'b n d -> b d', 'mean')

        out = self.norm(x)

        if not return_all_layers:
            return out

        return out, all_layers