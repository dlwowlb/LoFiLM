from einops import rearrange, repeat, reduce
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchaudio.transforms import Spectrogram


class Audiotransformer(nn.module):
    def __init__(accept_spec = False,
                    accept_spec_time_first = True,
                    spec_n_fft = 128,
                    spec_power = 2,
                    spec_win_length = 24,
                    spec_hop_length = None,
                    spec_pad = 0,
                    spec_center = True,
                    spec_pad_mode = 'reflect',
                    patch_dropout_prob = 0.25,
                 ):
        super().__init__()
        
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
        


        self.patch_dropout_prob = patch_dropout_prob
        
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
        
        height, width = x.shape[-2]
        

        
        return x