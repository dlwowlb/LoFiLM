import math
from functools import wraps, partial

##Test

import torch
import torch.nn.functional as F
from torch import nn, einsum

from torchaudio.transforms import Spectrogram, TimeStretch, FrequencyMasking, TimeMasking

#from audiolm_pytorch import AudioLM
#from audiolm_pytorch.utils import AudioConditionerBase

import torch.distributed as dist
#from musiclm_pytorch.distributed import AllGather

#from x_clip.tokenizer import tokenizer
#from vector_quantize_pytorch import ResidualVQ

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from beartype.typing import List, Optional, Tuple
from beartype import beartype

# functions

def exists(val):
    return val is not None

def first(it):
    return it[0]

def default(val, d):
    return val if exists(val) else d

def round_down_nearest_multiple(n, divisor):
    return n // divisor * divisor

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# decorators

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# tensor functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

# 2d sinusoidal positional embedding
# simple vit paper shows it is good enough compared to learned

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
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

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_hidden = int(dim * mult * 2 / 3)

    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, dim_hidden * 2, bias = False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim, bias = False)
    )

# attention

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

        # similarities

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

# transformer

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

        for attn, ff in self.layers:
            x = attn(x, rel_pos_bias = rel_pos_bias, mask = mask) + x
            x = ff(x) + x
            layers.append(x)

        if not return_all_layers:
            return x

        return x, torch.stack(layers[:-1])

# contrastive losses

class SoftmaxContrastiveLearning(nn.Module):
    def __init__(
        self,
        *,
        layers = 1,
        decoupled_contrastive_learning = False,
        init_temp = 10
    ):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(layers, 1, 1) * math.log(init_temp))
        self.decoupled_contrastive_learning = decoupled_contrastive_learning

        self.all_gather = AllGather(dim = 2)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, audio_latents, text_latents):
        if audio_latents.ndim == 2:
            audio_latents = rearrange(audio_latents, '... -> 1 ...')

        if text_latents.ndim == 2:
            text_latents = rearrange(text_latents, '... -> 1 ...')

        batch = audio_latents.shape[1]

        if self.all_gather.is_distributed:
            latents = torch.stack((audio_latents, text_latents))
            latents, _ = self.all_gather(latents)
            audio_latents, text_latents = latents

        sims = einsum('l i d, l j d -> l i j', audio_latents, text_latents)

        sims = sims * self.temperatures.exp()

        cosine_sims_exp = sims.exp()

        numerator = matrix_diag(cosine_sims_exp)

        if self.decoupled_contrastive_learning:
            eye = torch.eye(batch, device = self.device, dtype = torch.bool)
            cosine_sims_exp = cosine_sims_exp.masked_fill(eye, 0.)

        denominator_i = reduce(cosine_sims_exp, 'l i j -> l i', 'sum')
        denominator_j = reduce(cosine_sims_exp, 'l i j -> l j', 'sum')

        contrastive_loss = -log(numerator) + 0.5 * (log(denominator_i) + log(denominator_j))

        contrastive_loss = reduce(contrastive_loss, 'l n -> l', 'mean')
        return contrastive_loss.sum()

class SigmoidContrastiveLearning(nn.Module):
    """ https://arxiv.org/abs/2303.15343 """

    def __init__(
        self,
        *,
        layers = 1,
        init_temp = 10,
        init_bias = -10
    ):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(layers, 1, 1) * math.log(init_temp))
        self.bias = nn.Parameter(torch.ones(layers, 1, 1) * init_bias)

        self.all_gather = AllGather(dim = 1, all_reduce_grads = True)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, audio_latents, text_latents):
        device = self.device

        if audio_latents.ndim == 2:
            audio_latents = rearrange(audio_latents, '... -> 1 ...')

        if text_latents.ndim == 2:
            text_latents = rearrange(text_latents, '... -> 1 ...')

        text_latents, rank_sizes = self.all_gather(text_latents)

        n = text_latents.shape[1]

        sims = einsum('l i d, l j d -> l i j', audio_latents, text_latents)

        sims = sims * self.temperatures.exp() + self.bias

        labels = torch.eye(n, device = device)

        if exists(rank_sizes):
            labels_by_ranks = labels.split(rank_sizes.tolist(), dim = 0)
            labels = labels_by_ranks[dist.get_rank()]

        labels = 2 * rearrange(labels, 'i j -> 1 i j') - torch.ones_like(sims)

        return -F.logsigmoid(labels * sims).sum() / n

# Audio Spectrogram Transformer - https://arxiv.org/abs/2104.01778

def pair(t):
    return (t, t) if not isinstance(t, tuple) else t

class AudioSpectrogramTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        patch_size = 16,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        accept_spec = False,
        accept_spec_time_first = True,
        spec_n_fft = 128,
        spec_power = 2,
        spec_win_length = 24,
        spec_hop_length = None,
        spec_pad = 0,
        spec_center = True,
        spec_pad_mode = 'reflect',
        spec_aug_stretch_factor = 0.8,
        spec_aug_freq_mask = 80,
        spec_aug_time_mask = 80,
        patch_dropout_prob = 0.25
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.patch_size = pair(patch_size)
        patch_input_dim = self.patch_size[0] * self.patch_size[1]

        self.to_patch_tokens = Sequential(
            Rearrange('b (h p1) (w p2) -> b h w (p1 p2)', p1 = self.patch_size[0], p2 = self.patch_size[1]),
            nn.LayerNorm(patch_input_dim),
            nn.Linear(patch_input_dim, dim),
            nn.LayerNorm(dim)
        )

        self.accept_spec = accept_spec
        self.accept_spec_time_first = accept_spec_time_first

        self.spec = Spectrogram(
            n_fft = spec_n_fft,
            power = spec_power,
            win_length = spec_win_length,
            hop_length = spec_hop_length,
            pad = spec_pad,
            center = spec_center,
            pad_mode = spec_pad_mode
        )

        # SpecAugment - seems to be widely used in audio field https://arxiv.org/abs/1904.08779

        self.aug = torch.nn.Sequential(
            TimeStretch(spec_aug_stretch_factor, fixed_rate = True),
            FrequencyMasking(freq_mask_param = spec_aug_freq_mask),
            TimeMasking(time_mask_param = spec_aug_time_mask),
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

        self.norm = LayerNorm(dim)

        # patch dropout

        self.patch_dropout_prob = patch_dropout_prob

        # 2d dynamic positional bias

        mlp_hidden_dim = dim // 4

        self.dynamic_pos_bias_mlp = nn.Sequential(
            nn.Linear(2, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, heads),
            Rearrange('... i j h -> ... h i j')
        )

    def forward(
        self,
        x,
        force_no_patch_dropout = False,
        return_all_layers = False
    ):
        batch, device = x.shape[0], x.device
        assert (self.accept_spec and x.ndim == 3) or (not self.accept_spec and x.ndim == 2)

        if self.accept_spec and self.accept_spec_time_first:
            x = rearrange(x, 'b t f -> b f t')

        if not self.accept_spec:
            x = self.spec(x)

        if self.training:
            x = self.aug(x)

        # automatically crop if audio does not yield a 2d spectrogram that is divisible by patch sizes

        height, width = x.shape[-2:]
        patch_height, patch_width = self.patch_size

        rounded_height, rounded_width = map(lambda args: round_down_nearest_multiple(*args), ((height, patch_height), (width, patch_width)))

        if (height, width) != (rounded_height, rounded_width): # just keep printing to be annoying until it is fixed
            print_once(f'spectrogram yielded shape of {(height, width)}, but had to be cropped to {(rounded_height, rounded_width)} to be patchified for transformer')

        x = x[..., :rounded_height, :rounded_width]

        # to patches

        x = self.to_patch_tokens(x)

        # get number of patches along height and width

        _, num_patch_height, num_patch_width, _ = x.shape

        # get 2d relative positions

        grid = torch.stack(torch.meshgrid(
            torch.arange(num_patch_height, device = device),
            torch.arange(num_patch_width, device = device)
        , indexing = 'ij'), dim = -1)

        grid = rearrange(grid, '... c -> (...) c')

        # 2d sinusoidal positional embedding

        x = x + posemb_sincos_2d(x)

        x = rearrange(x, 'b ... c -> b (...) c')

        # patch dropout

        if self.training and self.patch_dropout_prob > 0. and not force_no_patch_dropout:
            n, device = x.shape[1], x.device

            batch_indices = torch.arange(batch, device = device)
            batch_indices = rearrange(batch_indices, '... -> ... 1')
            num_patches_keep = max(1, int(n * (1 - self.patch_dropout_prob)))
            patch_indices_keep = torch.randn(batch, n, device = device).topk(num_patches_keep, dim = -1).indices

            x = x[batch_indices, patch_indices_keep]

            grid = repeat(grid, '... -> b ...', b = batch)
            grid = grid[batch_indices, patch_indices_keep]

        # 2d relative positional bias

        rel_dist = rearrange(grid, '... i c -> ... i 1 c') - rearrange(grid, '... j c -> ... 1 j c')
        rel_pos_bias = self.dynamic_pos_bias_mlp(rel_dist.float())

        # attention, what else

        x, all_layers = self.transformer(x, rel_pos_bias = rel_pos_bias, return_all_layers = True)

        # final global average and norm (most recent papers show this is superior to CLS token)

        x = reduce(x, 'b n d -> b d', 'mean')

        out = self.norm(x)

        if not return_all_layers:
            return out

        return out, all_layers

'''
if __name__ == '__main__':
    audio_transformer = AudioSpectrogramTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64,
    spec_n_fft = 128,
    spec_win_length = 24,
    spec_aug_stretch_factor = 0.8
)
'''