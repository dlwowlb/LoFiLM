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



    def forward(self, x):
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
        

        #patch dropout
        if self.training and self.patch_dropout_prob > 0. and not force_no_patch_dropout:
            n, device = x.shape[1], x.device

            batch_indices = torch.arange(batch, device = device)
            batch_indices = rearrange(batch_indices, '... -> ... 1')
            num_patches_keep = max(1, int(n * (1 - self.patch_dropout_prob)))
            patch_indices_keep = torch.randn(batch, n, device = device).topk(num_patches_keep, dim = -1).indices

            x = x[batch_indices, patch_indices_keep]

            grid = repeat(grid, '... -> b ...', b = batch)
            grid = grid[batch_indices, patch_indices_keep]
        
        return x