from Transformer import AudioTransformer, TextTransformer
from Transformer.AudioTransformer import l2norm, default, LayerNorm
from Distributed import AllGather


import math
from functools import wraps, partial

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat, reduce

from beartype.typing import List, Optional, Tuple
from beartype import beartype

from vector_quantize_pytorch import ResidualVQ 


#전체 레이어 중 일부 레이어만 고르게 선택할 수 있도록 인덱스를 가져오는 함수
#layers: 선택할 레이어 수, total_layers: 전체 레이어 수
def interspersed_indices(layers, total_layers):
    assert total_layers >= layers
    step = total_layers / layers
    return (torch.arange(0, layers) * step).floor().long()


#CLIP 모델의 InfoNCE Loss와 유사(양성 샘플 유사도 높이고, 음성 샘플 유사도 낮추는 방향으로 학습)
#오디오와 텍스트 간 유사도를 계산
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
        #분산학습시 AllGather 사용
        self.all_gather = AllGather(dim = 2)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, audio_latents, text_latents):
        # 임베딩이 (batch, dim)이면 (1, batch, dim)으로 차원 추가
        if audio_latents.ndim == 2: 
            audio_latents = rearrange(audio_latents, '... -> 1 ...')
        if text_latents.ndim == 2:
            text_latents = rearrange(text_latents, '... -> 1 ...')

        #배치 크기 가져오기기
        batch = audio_latents.shape[1]

        #분산 환경에서 모든 프로세스(rank)의 텍스트 임베딩을 **가로 방향(batch 차원)**으로 모아줌
        if self.all_gather.is_distributed:
            latents = torch.stack((audio_latents, text_latents))
            latents, _ = self.all_gather(latents)
            audio_latents, text_latents = latents #shape는 (layers, batch, dim)이 됨

        #레이어별로 모든 오디오 vs 모든 텍스트 쌍에 대한 dot product를 구합니다.
        sims = einsum('l i d, l j d -> l i j', audio_latents, text_latents)

        #온도 파라미터가 크면 유사도 차이가 더 크게 반영되어 분포가 sharp해진다.
        #위에서 log를 취해주었기 때문에 exp를 취해줌
        sims = sims * self.temperatures.exp()

        #소프트맥스를 취해주어 유사도를 확률로 변환
        cosine_sims_exp = sims.exp()

        #대각 성분만 가져옴(양성 항)(e.g. (audio1, text1), (audio2, text2), ...)
        numerator = matrix_diag(cosine_sims_exp)

        #loss 함수에서 양성 음성 분리
        #대각 성분을 0으로 만들어줌(불리언 마스크로)
        #대각선(양성 위치)를 0으로 만들어서, loss function에서 분모 계산시 양성 항이 제외되도록
        #더욱 안정적 학습 가능
        if self.decoupled_contrastive_learning:
            eye = torch.eye(batch, device = self.device, dtype = torch.bool)
            cosine_sims_exp = cosine_sims_exp.masked_fill(eye, 0.)
        #오디오 i에 대해 모든 텍스트 j에 대한 유사도 합
        #텍스트 j에 대한 모든 오디오 i에 대한 유사도 합
        denominator_i = reduce(cosine_sims_exp, 'l i j -> l i', 'sum')
        denominator_j = reduce(cosine_sims_exp, 'l i j -> l j', 'sum')

        #최종적으로 Contrastive Loss 계산
        #양성 항은 -log 취해주고, 음성 항은 log 취하고 0.5를 곱해줌(양성, 음성 항이 모두 고려하기 위함)
        contrastive_loss = -log(numerator) + 0.5 * (log(denominator_i) + log(denominator_j))

        #배치 차원에 대해 평균 취함(각 레이어별로 배치 평균 로스)
        contrastive_loss = reduce(contrastive_loss, 'l n -> l', 'mean')
        return contrastive_loss.sum() #최종적으로 레이버 차원을 전부 합산(단일 스칼라 로스)

#레이어별로 임베딩을 가져와서 Contrastive Loss 계산
class MultiLayerContrastiveLoss(nn.Module):
    def __init__(
        self,
        *,
        audio_dim,
        text_dim,
        dim_latent,
        layers,
        decoupled_contrastive_learning = False,
        sigmoid_contrastive_loss = False
    ):
        super().__init__()
        self.layers = layers

        #LayerNorm에서 scale=False로 설정하여, gamma 파라미터를 사용하지 않음
        #대신 레이어별로 gamma 파라미터를 사용하기 위해 nn.Parameter로 선언
        self.audio_norm = LayerNorm(audio_dim, scale = False)
        self.audio_gamma = nn.Parameter(torch.ones(layers, 1, audio_dim))
        
        #오디오 임베딩 -> latent space로 projection
        self.audio_latent_weight = nn.Parameter(torch.randn(layers, audio_dim, dim_latent))
        self.audio_latent_bias = nn.Parameter(torch.randn(layers, 1, dim_latent))

        #텍스트도 동일
        self.text_norm = LayerNorm(text_dim, scale = False)
        self.text_gamma = nn.Parameter(torch.ones(layers, 1, text_dim))
        self.text_latent_weight = nn.Parameter(torch.randn(layers, text_dim, dim_latent))
        self.text_latent_bias = nn.Parameter(torch.randn(layers, 1, dim_latent))

        #sigmoid_contrastive_loss가 True일 경우 SigmoidContrastiveLearning을 사용하고,
        #아닐 경우 SoftmaxContrastiveLearning을 사용
        klass = SigmoidContrastiveLearning if sigmoid_contrastive_loss else partial(SoftmaxContrastiveLearning, decoupled_contrastive_learning = decoupled_contrastive_learning)
        self.contrast = klass(layers = layers)

    def forward(self, *, audio_layers, text_layers):
        device, batch = audio_layers.device, audio_layers.shape[1]

        #시퀀스 길이로 평균 내어서 (layers, batch, dim)
        #시퀀스 단위 글로벌 평균 풀링
        #LayerNorm을 거친다음, 레이어별 스케일 파라미터(audio_gamma)를 곱해줌
        audio_gap = reduce(audio_layers, 'l b n d -> l b d', 'mean')
        audio_embeds = self.audio_norm(audio_gap) * self.audio_gamma

        #(layers, batch, dim) × (layers, dim, dim_latent) → (layers, batch, dim_latent)
        #레이어(l)별로 따로 곱해짐
        audio_latents = einsum('l b d, l d e -> l b e', audio_embeds, self.audio_latent_weight) + self.audio_latent_bias
        audio_latents = l2norm(audio_latents)

        #텍스트는 CLS 토큰만 사용(BERT)
        text_cls_tokens = text_layers[:, :, 0]

        #나머지는 동일
        text_embeds = self.text_norm(text_cls_tokens) * self.text_gamma
        text_latents = einsum('l b d, l d e -> l b e', text_embeds, self.text_latent_weight) + self.text_latent_bias
        text_latents = l2norm(text_latents)

        #레이어별로 Contrastive Loss 계산
        return self.contrast(audio_latents, text_latents)


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

        #레이어별 Contrastive Loss 계산을 위한 MultiLayerContrastiveLoss
        self.multi_layer_contrastive_learning = None

        #Hierarchical Contrastive Loss 사용 시
        if hierarchical_contrastive_loss:
            # hierarchical_contrastive_loss_layers가 None이 아니면 사용
            # 아니면 min(audio_transformer.depth, text_transformer.depth) - 1
            num_layers = default(hierarchical_contrastive_loss_layers, min(audio_transformer.depth, text_transformer.depth) - 1)
            
            #레이어 수가 0 이하면 에러 발생
            assert num_layers > 0

            #depth가 12, num_layers가 4면 2,5,8,11번째 골고루 선택
            self.register_buffer('text_layers_indices', interspersed_indices(num_layers, text_transformer.depth))
            self.register_buffer('audio_layers_indices', interspersed_indices(num_layers, audio_transformer.depth))

            #멀티 레이어 대조학습 
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
        #샘플별 유사도 계산
        if return_similarities:
            return einsum('i d, i d -> i', audio_latents, text_latents)

        #위와는 다르게 샘플별 유사도가 아닌 쌍별 유사도를 계산
        if return_pairwise_similarities:
            cosine_sim = einsum('i d, j d -> i j', audio_latents, text_latents)
            return cosine_sim


        cl_loss = self.contrast(audio_latents, text_latents)

        if not exists(self.multi_layer_contrastive_learning):
            return cl_loss

        #위의 버퍼로부터 가져온 인덱스를 사용하여 레이어별로 Contrastive Loss 계산
        audio_layers = audio_layers[self.audio_layers_indices]
        text_layers = text_layers[self.text_layers_indices]

        #여러 레이어의 임베딩 모두 활용해 대조학습, from ViCHA paper https://arxiv.org/abs/2208.13628

        hierarchical_cl_loss = self.multi_layer_contrastive_learning(
            audio_layers = audio_layers,
            text_layers = text_layers
        )
        #레이어별로 계산된 Contrastive Loss와 전체 Contrastive Loss를 합침
        #최종 레이어 표현과 중간 레이어들에서 골고루 골라서 학습
        return cl_loss + hierarchical_cl_loss

#추상 베이스 클래스
class AudioConditionerBase(nn.Module):
    pass
    
#semantic, coarse, fine quantization
class MuLaNEmbedQuantizer(AudioConditionerBase):
    @beartype
    def __init__(
        self,
        mulan: MuLaN,
        conditioning_dims: Tuple[int, ...],
        rq_num_quantizers = 8,
        rq_ema_decay = 0.9,
        codebook_size = 1024,
        namespaces: Tuple[str, ...] = ('semantic', 'coarse', 'fine'),

    ):
        super().__init__()
        self.mulan = mulan
        assert len(namespaces) > 0
        
        self.namespaces = namespaces
        self.conditioning_dims = conditioning_dims

        assert len(conditioning_dims) == len(namespaces), 'number of conditioning dimensions must be equal to number of namespaces'

        dim = mulan.dim_latent


        #이미 정의된 코드북 코드 사용
        #Residual VQ는 한번에 양자화하는 대신 여러 단계로 나눠서 양자화
        self.rq = ResidualVQ(
            dim = dim,
            num_quantizers = rq_num_quantizers,
            codebook_size = codebook_size,
            decay = rq_ema_decay,
            commitment_weight = 0,    # only use EMA to update codebooks
            kmeans_init = True,
            threshold_ema_dead_code = 2,
            quantize_dropout = False  # no quantize dropout
        )

        self.dim = dim
        self.num_codebooks = rq_num_quantizers

        #딕셔너리 초기화해서 저장
        self.cond_embeddings = nn.ParameterDict({})

        for namespace, conditioning_dim in zip(namespaces, conditioning_dims):
            cond_embeddings = nn.Parameter(torch.randn(rq_num_quantizers, codebook_size, conditioning_dim))
            nn.init.normal_(cond_embeddings, std = 0.02)

            self.cond_embeddings[namespace] = cond_embeddings

        self.set_default_namespace(namespaces[0])

    def parameters(self):
        return self.cond_embeddings.parameters()

    def set_default_namespace(self, namespace):
        self._default_namespace = namespace

    def forward(self,wavs = None,texts = None,namespace = None):
        #오디오나 텍스트 중 하나만 존재해야 함
        assert exists(wavs) ^ exists(texts)

        #namespace가 없으면 default namespace 사용
        #namespace가 존재하지 않으면 에러 발생
        namespace = default(namespace, self._default_namespace)
        assert namespace in self.namespaces, f'namespace {namespace} not found'
        
        #코드워드(실제 코드북 벡터)
        cond_embeddings = self.cond_embeddings[namespace]

        with torch.no_grad():
            self.mulan.eval()

            # sound and language live in joint embedding space because of contrastive learning

            if exists(wavs):
                latents = self.mulan.get_audio_latents(wavs)
            elif exists(texts):
                latents = self.mulan.get_text_latents(texts)

        #양자화 수행해서 인덱스만 가져옴
        _, indices, _ = self.rq(latents) #latent를 이미 정의된 양자화 기법으로 이산화

        batch, num_codebooks, dim = indices.shape[0], self.num_codebooks, cond_embeddings.shape[-1]

        cond_embeddings = repeat(cond_embeddings, 'q c d -> b q c d', b = batch)
        indices = repeat(indices, 'b q -> b q 1 d', q = num_codebooks, d = dim)

        cond_embeddings = cond_embeddings.gather(2, indices)
        return rearrange(cond_embeddings, 'b q 1 d -> b q d')
    


class MusicLM(nn.Module):
    @beartype
    def __init__(
        self,
        audio_lm: AudioLM,
        mulan_embed_quantizer: MuLaNEmbedQuantizer
    ):
        super().__init__()
        assert not exists(audio_lm.audio_conditioner), 'mulan must not have been passed into AudioLM. it will be managed externally now, embedding the text into the joint embedding space for text-to-audio synthesis'

        self.mulan_embed_quantizer = mulan_embed_quantizer
        self.audio_lm = audio_lm

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def forward(
        self,
        text: str,
        num_samples = 1,
        **audio_lm_kwargs
    ):
        self.eval()

        texts = tokenizer.tokenize([text]).to(self.device)

        text_embeds = self.mulan_embed_quantizer(texts = texts)

        # unable to deal with variable lengthed audio for now

        samples = []

        for _ in range(num_samples):
            music = self.audio_lm(text_embeds = text_embeds, **audio_lm_kwargs)
            samples.append(music)

        # if one sample, just return it

        if num_samples == 1:
            return first(samples)

        mulan = self.mulan_embed_quantizer.mulan

        # get the one with the highest similarity score, of all the samples

        sims = torch.cat([mulan(texts = texts, wavs = music, return_similarities = True) for music in samples], dim = 0)
        top_matching_index = sims.topk(1, dim = 0).indices.item()

        return samples[top_matching_index]