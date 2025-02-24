from Transformer import AudioTransformer, TextTransformer
from Transformer.AudioTransformer import l2norm
from Distributed import AllGather


import math
from functools import wraps, partial

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat, reduce

from beartype.typing import List, Optional, Tuple
from beartype import beartype



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

#Contrastive Learning을 sigmoid + 로지스틱 로스 방식
class SigmoidContrastiveLearning(nn.Module):

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

        #layer차원 추가
        if audio_latents.ndim == 2:
            audio_latents = rearrange(audio_latents, '... -> 1 ...')
        if text_latents.ndim == 2:
            text_latents = rearrange(text_latents, '... -> 1 ...')

        #분산 학습(DDP) 시, 모든 프로세스(rank)의 텍스트 임베딩을 **가로 방향(batch 차원)**으로 모아줌
        text_latents, rank_sizes = self.all_gather(text_latents)

        n = text_latents.shape[1]

        #(layers, batch, dim) × (layers, n, dim) → (layers, batch, n)
        # 레이어별로 모든 오디오 vs 모든 텍스트 쌍에 대한 dot product를 구합니다.
        sims = einsum('l i d, l j d -> l i j', audio_latents, text_latents)

        #위에서 구한 유사도에 온도 곱(온도는 위에서 log를 취해주었기 때문에 exp를 취해줌)
        #온도(temp)가 커지면 유사도 값의 범위가 확대되어 학습 신호가 강해질 수 있습니다. bias는 일종의 기준점 역할.
        sims = sims * self.temperatures.exp() + self.bias


        labels = torch.eye(n, device = device)

        #DDP 환경에서 각 rank는 자기 배치에 해당하는 라벨만 필요하므로, labels.split(...) 후 현재 rank에 맞는 부분을 취함.
        if exists(rank_sizes):
            labels_by_ranks = labels.split(rank_sizes.tolist(), dim = 0)
            labels = labels_by_ranks[dist.get_rank()]

        #[0, 1]로 표현된 라벨을 **[-1, +1]**로 변환. 이는 BCEWithLogitsLoss를 사용하기 위함.
        labels = 2 * rearrange(labels, 'i j -> 1 i j') - torch.ones_like(sims)

        #labels * sims가 양성(1)에서는 큰 양수, 음성(-1)에서는 큰 음수가 되도록 학습.
        return -F.logsigmoid(labels * sims).sum() / n

class MuLaN(nn.Module):
    @beartype
    def __init__(
        self,
        audio_transformer: AudioTransformer,
        text_transformer: TextTransformer,
        dim_latent = 128,                       # they use 128
        decoupled_contrastive_learning = True,  # think this was used, make it optional
        hierarchical_contrastive_loss = False,
        hierarchical_contrastive_loss_layers = None,
        sigmoid_contrastive_loss = False
    ):
        super().__init__()
        self.dim_latent = dim_latent

        self.audio = audio_transformer
        self.text = text_transformer


        self.text_to_latents = nn.Linear(self.text.dim, dim_latent)
        self.audio_to_latents = nn.Linear(self.audio.dim, dim_latent)

        # sigmoid_contrastive_loss가 True일 경우 SigmoidContrastiveLearning을 사용하고, 
        # 아닐 경우 SoftmaxContrastiveLearning을 사용
        # 여기서 partial은 def some_function():
                            #return SoftmaxContrastiveLearning(decoupled_contrastive_learning=True)
        klass = SigmoidContrastiveLearning if sigmoid_contrastive_loss else partial(
            SoftmaxContrastiveLearning,
            decoupled_contrastive_learning = decoupled_contrastive_learning
        )
        self.contrast = klass()

        self.multi_layer_contrastive_learning = None

        if hierarchical_contrastive_loss:
            num_layers = default(hierarchical_contrastive_loss_layers, min(audio_transformer.depth, text_transformer.depth) - 1)
            assert num_layers > 0

            self.register_buffer('text_layers_indices', interspersed_indices(num_layers, text_transformer.depth))
            self.register_buffer('audio_layers_indices', interspersed_indices(num_layers, audio_transformer.depth))

            self.multi_layer_contrastive_learning = MultiLayerContrastiveLoss(
                audio_dim = self.audio.dim,
                text_dim = self.text.dim,
                dim_latent = dim_latent,
                layers = num_layers,
                decoupled_contrastive_learning = decoupled_contrastive_learning,
                sigmoid_contrastive_loss = sigmoid_contrastive_loss
            )

    def get_audio_latents(self,wavs,return_all_layers = False):

        #오디오 Transformer를 사용하여 오디오 잠재 공간을 가져옴
        audio_embeds, audio_layers = self.audio(wavs, return_all_layers = True)
        audio_latents = self.audio_to_latents(audio_embeds) # 그냥 nn.parameter로 선언된 것을 통과시키는 것
        out = l2norm(audio_latents)

        if not return_all_layers:
            return out

        return out, audio_layers

    @beartype
    def get_text_latents(self,texts = None,raw_texts: Optional[List[str]] = None,return_all_layers = False):
        
        #텍스트 Transformer를 사용하여 텍스트 잠재 공간을 가져옴
        #이때 토큰화된 Text나 토큰화가 아직 안된 Raw_text가 있을 수 있음
        text_embeds, text_layers = self.text(texts, raw_texts = raw_texts, return_all_layers = True)
        text_latents = self.text_to_latents(text_embeds)
        out = l2norm(text_latents)

        if not return_all_layers:
            return out

        return out, text_layers

    @beartype
    def forward(
        self,
        wavs,
        texts = None,
        raw_texts: Optional[List[str]] = None,
        return_latents = False,
        return_similarities = False,
        return_pairwise_similarities = False
    ):
        
        #배치 크기와 디바이스 가져오기
        batch, device = wavs.shape[0], wavs.device

        #오디오와 텍스트 잠재 공간 가져오기, 중간 레이어도 가져오기기
        audio_latents, audio_layers = self.get_audio_latents(wavs, return_all_layers = True)
        text_latents, text_layers = self.get_text_latents(texts, raw_texts = raw_texts, return_all_layers = True)


        if return_latents:
            return audio_latents, text_latents

        #오디오 임베딩과 텍스트 임베딩의 dot product 계산(유사도 구하기)
        #einsum('i d, i d -> i')는 (batch, dim) × (batch, dim) → (batch,) 형태로
        #샘플별 유사도 계산산
        if return_similarities:
            return einsum('i d, i d -> i', audio_latents, text_latents)

        #위와는 다르게 샘플별 유사도가 아닌 쌍별 유사도를 계산
        if return_pairwise_similarities:
            cosine_sim = einsum('i d, j d -> i j', audio_latents, text_latents)
            return cosine_sim


        cl_loss = self.contrast(audio_latents, text_latents)

        if not exists(self.multi_layer_contrastive_learning):
            return cl_loss

        audio_layers = audio_layers[self.audio_layers_indices]
        text_layers = text_layers[self.text_layers_indices]

        # whether to do cl loss across all layers, from ViCHA paper https://arxiv.org/abs/2208.13628

        hierarchical_cl_loss = self.multi_layer_contrastive_learning(
            audio_layers = audio_layers,
            text_layers = text_layers
        )

        return cl_loss + hierarchical_cl_loss