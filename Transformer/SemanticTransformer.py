from utils import *
from functools import partial, wraps

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

import torch.nn.functional as F
import torch
from torch import nn
from torchaudio.functional import resample

from pathlib import Path
import fairseq
from tqdm import tqdm

import logging
logging.root.setLevel(logging.ERROR)

class SemanticTransformer(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        dim,
        depth,
        num_semantic_tokens,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        t5_name = DEFAULT_T5_NAME,
        cond_dim = None,
        has_condition = False,
        audio_text_condition = False,
        cond_as_self_attn_prefix = False,
        cond_drop_prob = 0.5,
        grad_shrink_alpha = 0.1,
        rel_pos_bias = True,
        flash_attn = False,
        **kwargs
    ):
        super().__init__()
        rel_pos_bias = rel_pos_bias and not flash_attn

        self.num_semantic_tokens = num_semantic_tokens

        if audio_text_condition:
            has_condition = True
            cond_dim = default(cond_dim, dim)

        self.has_condition = has_condition
        self.embed_text = partial(t5_encode_text, name = t5_name)
        self.cond_drop_prob = cond_drop_prob

        self.start_token = nn.Parameter(torch.randn(dim))

        self.semantic_embedding = nn.Embedding(num_semantic_tokens + 1, dim)
        self.eos_id = num_semantic_tokens

        #text_dim과 dim이 같지 않으면 텍스트 임베딩 차원을 dim으로 변환
        #만약 같으면 nn.Identity()로 변환(입력값으로 그냥냥)
        text_dim = default(cond_dim, get_encoded_dim(t5_name))
        self.proj_text_embed = nn.Linear(text_dim, dim, bias = False) if text_dim != dim else nn.Identity()

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            cross_attend = has_condition and not cond_as_self_attn_prefix,
            cond_as_self_attn_prefix = cond_as_self_attn_prefix,
            grad_shrink_alpha = grad_shrink_alpha,
            rel_pos_bias = rel_pos_bias,
            flash_attn = flash_attn,
            **kwargs
        )

        #토큰 추가해서 로짓 생성
        self.to_logits = nn.Linear(dim, num_semantic_tokens + 1)

    @property
    def device(self):
        return next(self.parameters()).device

    def load(self, path):
        # Return pkg so that if this function gets called from within a Trainer function call,
        # the trainer can also access the package loaded from the checkpoint.
        device = self.device
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = device)
        # check version
        if 'version' in pkg and version.parse(pkg['version']) < version.parse(__version__):
            print(f'model was trained on older version {pkg["version"]} of audiolm-pytorch')
        self.load_state_dict(pkg['model'])
        return pkg

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3,
        kv_cache = None,
        return_kv_cache = False,
        **kwargs
    ):
        kv_cache = iter(default(kv_cache, [])) #kv캐시 있으면 이터레이터, 없으면 빈 리스트로 이터레이터
        new_kv_caches = []

        #next(kv_cache, None)는 kv_cache의 값을 차례대로 반환, 없으면 None 반환
        logits, new_kv_cache = self.forward(*args, cond_drop_prob = 0., 
                                            kv_cache = next(kv_cache, None), 
                                            return_kv_cache = True, **kwargs)
        new_kv_caches.append(new_kv_cache)

        #cond_scale이 1이면 그대로 반환
        if cond_scale == 1 or not self.has_condition:
            if not return_kv_cache:
                return logits

            return logits, torch.stack(new_kv_caches)

        #null 조건은 마스킹 비율을 1로 설정해서 unconditional로 만듬
        #unconditional이랸 조건(텍스트, 이미지 등)이 없이 학습된 내재 지식을 바탕으로 예측
        null_logits, null_new_kv_cache = self.forward(*args, cond_drop_prob = 1., 
                                                      kv_cache = next(kv_cache, None), 
                                                      return_kv_cache = True, **kwargs)
        new_kv_caches.append(null_new_kv_cache)

        #조건이 적용된 예측값과 조건이 제거된 예측값의 차이에 cond_scale을 곱함
        #classifier-free guidance를 위한 방법
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if not return_kv_cache:
            return scaled_logits

        return scaled_logits, torch.stack(new_kv_caches)

    @beartype
    def forward(
        self,
        *,
        ids = None,
        return_loss = False,
        text: list[str] | None = None,
        text_embeds = None,
        self_attn_mask = None,
        cond_drop_prob = None,
        unique_consecutive = None,
        kv_cache = None,
        return_kv_cache = False
    ):
        device = self.device

        b = ids.shape[0]

        has_text = exists(text) or exists(text_embeds)
        assert not (self.has_condition ^ has_text)

        #텍스트 임베딩은 없고 텍스트만 있을때 텍스트 임베딩 생성
        #T5를 통해 텍스트를 인코딩하여 임베딩 생성
        text_mask = None
        if not exists(text_embeds) and exists(text):
            with torch.inference_mode():
                text_embeds = self.embed_text(text, output_device = device)
                text_mask = torch.any(text_embeds != 0, dim = -1)

        #텍스트 임베딩이 완료되면 차원확인 후 변환
        if exists(text_embeds):
            text_embeds = self.proj_text_embed(text_embeds)

        #cond_drop_prob가 있으면 그대로 사용 아니면 self.cond_drop_prob 사용
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        #cond_drop_prob가 0이 아니면 cond_drop_prob만큼의 확률로 마스킹
        if exists(text_mask) and cond_drop_prob > 0:
            keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device = device) #마스크 만들고
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask #마스크 적용

        #로스를 반환해야하면 label을 ids로 그대로 활용(teacher forcing 방식)
        #ids는 마지막 토큰을 제외한 것 -> 다음 토큰을 예측하게 하기 위함
        #[a,b,c,d]라고 했을때 라벨은 [a,b,c,d] 입력은 [a,b,c] 
        #[a,b,c]를 입력받으면 d를 예측하도록
        if return_loss:
            labels, ids = ids.clone(), ids[:, :-1]

        #임베딩 벡터로 변환(패딩 포함)
        #nn.embedding이고 마스크 위치에 0(혹은 다른 값) 넣기  
        tokens = get_embeds(self.semantic_embedding, ids)

        #start token은 무작위 값(학습 가능함)
        #(batch_size, 1, d)로 만들어서 토큰에 추가
        start_tokens = repeat(self.start_token, 'd -> b 1 d', b = ids.shape[0])

        #start token을 토큰에 추가(시퀀스 차원으로)
        tokens = torch.cat((start_tokens, tokens), dim = 1)

        #마스크가 있으면 좌측에 1 우측에 0 만큼 True값 패딩 추가
        if exists(self_attn_mask):
            self_attn_mask = F.pad(self_attn_mask, (1, 0), value = True)

        #utils.py에 있는 Transformer 클래스의 forward 함수 호출
        #key value cache가 있으면 사용
        #key value cache는 한 토큰씩 생성할 때 이전 토큰들에서 이미 계산된 key, value 재사용용
        tokens, kv_cache = self.transformer(tokens, context = text_embeds, self_attn_mask = self_attn_mask, context_mask = text_mask, kv_cache = kv_cache, return_kv_cache = True)
        logits = self.to_logits(tokens)


        if not return_kv_cache:
            return logits

        return logits, kv_cache

class FairseqVQWav2Vec(nn.Module):
    """
    checkpoint path can be found at https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md#vq-wav2vec
    specifically download the kmeans model for now

    $ wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt
    """

    def __init__(
        self,
        checkpoint_path, #체크포인트 경로
        target_sample_hz = 24000, #목표 샘플링 주파수
        seq_len_multiple_of = None #오디오 시퀀스 길이를 특정 배수로 자르기기
    ):
        super().__init__()
        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = seq_len_multiple_of

        path = Path(checkpoint_path)
        assert path.exists(), f'path {checkpoint_path} does not exist'

        checkpoint = torch.load(checkpoint_path)
        load_model_input = {checkpoint_path: checkpoint}
        #위의 wget에서 wav2vec모델 사용한다고 했고
        #fairseq에 있는 여러 모델 앙상블, task, 모델을 로드
        model, *_ = fairseq.checkpoint_utils.load_model_ensemble_and_task(load_model_input)

        self.model = model[0]
        #VQ-Wav2vec pretrained 모델을 eval모드로 설정해서 코드북만 가져옴
        self.model.eval()

        #모델이 유효한지 확인
        #VQ-Wave2Vec에서 벡터 양자화 담당, 오디오 임베딩을 코드북 인덱스로 변환
        #VQ-Wave2Vec 모델이 유효한지 확인
        assert hasattr(self.model, 'vector_quantizer') and hasattr(self.model.vector_quantizer, 'embedding'), 'the vq wav2vec model does not seem to be valid'

    @property
    def groups(self):
        return self.model.vector_quantizer.groups

    @property
    def downsample_factor(self):
        # todo: double check architecture
        return 80

    @property
    def codebook_size(self):
        return self.model.vector_quantizer.embedding.shape[0]

    @torch.inference_mode()
    def forward(
        self,
        wav_input,
        flatten = True,
        input_sample_hz = None
    ):
        #오디오 입력을 목표 샘플링 주파수로 변환
        if exists(input_sample_hz):
            wav_input = resample(wav_input, input_sample_hz, self.target_sample_hz)

        #오디오 시퀀스 길이를 특정 배수로 자르기
        if exists(self.seq_len_multiple_of):
            wav_input = curtail_to_multiple(wav_input, self.seq_len_multiple_of)

        #오디오 입력을 모델에 통과시켜 임베딩 추출
        embed = self.model.feature_extractor(wav_input)

        #추출된 임베딩을 벡터 양자화된 결과로 코드북 인덱스 얻음
        _, codebook_indices = self.model.vector_quantizer.forward_idx(embed)

        if not flatten:
            return codebook_indices

        return rearrange(codebook_indices, 'b ... -> b (...)')


#시멘틱 토큰 생성하기 위함
class SemanticTransformerWrapper(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        transformer: SemanticTransformer,
        wav2vec: FairseqVQWav2Vec | HubertWithKmeans | None = None, #세 가지 타입 중 하나, 기본은 None
        audio_conditioner: AudioConditionerBase | None = None,
        pad_id = -1,
        unique_consecutive = True,
        mask_prob = 0.15
    ):
        super().__init__()
        self.wav2vec = wav2vec
        self.transformer = transformer
        self.to(transformer.device)
        self.audio_conditioner = audio_conditioner

        #
        assert not (exists(audio_conditioner) and not transformer.has_condition), 'if conditioning on audio embeddings from mulan, transformer has_condition must be set to True'

        assert not exists(self.wav2vec) or self.wav2vec.codebook_size == transformer.num_semantic_tokens, f'num_semantic_tokens on SemanticTransformer must be set to {self.wav2vec.codebook_size}'

        self.unique_consecutive = unique_consecutive
        self.pad_id = pad_id
        self.eos_id = transformer.eos_id
        self.mask_prob = mask_prob

    @property
    def device(self):
        return next(self.parameters()).device

    def embed_text(self, text):
        return self.transformer.embed_text(text, output_device = self.device)

    #시맨틱 토큰 생성
    @eval_decorator #데코레이터
    @torch.inference_mode()
    @beartype
    def generate(
        self,
        *,
        max_length,
        text: list[str] | None = None,
        text_embeds = None,
        prime_wave = None,
        prime_wave_input_sample_hz = None,
        prime_ids = None,
        batch_size = 1,
        cond_scale = 3,
        filter_thres = 0.9,
        temperature = 1.,
        use_kv_cache = True,
        include_eos_in_output = True,  # if doing hierarchical sampling, eos must be kept for an easy time
        **kwargs
    ):
        device = self.device

        # derive wav2vec ids from the input wave

        #prime_wave라는게 주어지면 wave2vec을 통해 ids를 생성
        if exists(prime_wave):
            assert not exists(prime_ids)
            assert exists(self.wav2vec)
            #위에서 정의한 코드북 인덱스 생성
            ids = self.wav2vec(
                prime_wave,
                flatten = False,
                input_sample_hz = prime_wave_input_sample_hz
            )
        elif exists(prime_ids):
            ids = prime_ids
        else:
            ids = torch.empty((batch_size, 0), dtype = torch.long, device = device)

        #중복제거 및 패딩 추가
        if self.unique_consecutive:
            ids = batch_unique_consecutive(ids, pad_value = self.pad_id)

        # derive joint audio-text embeddings if needed

        #오디오 컨디셔너 존재하고 prime_wave가 존재하면 텍스트와 텍스트 임베딩은 없어야함
        if exists(self.audio_conditioner) and exists(prime_wave):
            assert not exists(text) and not exists(text_embeds)
            #텍스트, 텍스트 임베딩 없는거 확인했으면 시맨틱 텍스트 임베딩 생성
            #실제로 생성한건 아닌것처럼 보임, base class 사용 <- 확인 바람
            text_embeds = self.audio_conditioner(wavs = prime_wave, namespace = 'semantic')

        #텍스트나 텍스트 임베딩이 있으면(has_text가 True이면면) has_condition이 True여야함
        has_text = exists(text) or exists(text_embeds)
        assert not (self.transformer.has_condition ^ has_text)

        #텍스트 임베딩이 없고 텍스트만 있으면 텍스트 임베딩 생성
        if not exists(text_embeds) and exists(text):
            with torch.inference_mode():
                text_embeds = self.transformer.embed_text(text, output_device = device)

        # start length and get running id output
        batch = ids.shape[0]
        start_length = ids.shape[-1]
        sample_semantic_ids = ids.clone()

        #시퀸스에서 패딩 토큰이 아닌 토큰의 개수, 유효 토큰의 마지막 인덱스 구함
        #패딩이 아니면 1, 패딩이면 0으로 표시
        #마지막 유효한 토큰의 인덱스(배치별로 패딩이 아닌 토큰의 개수)
        last_logit_indices = (ids != self.pad_id).sum(dim = -1).long()

        # kv cache

        kv_cache = None
        logits = None

        # sample from transformer

        for ind in tqdm(range(start_length, max_length), desc = 'generating semantic'):

            #조건 스케일링을 적용하여 logits 생성
            new_logits, new_kv_cache = self.transformer.forward_with_cond_scale(
                ids = sample_semantic_ids,
                text_embeds = text_embeds,
                cond_scale = cond_scale,
                kv_cache = kv_cache,
                return_kv_cache = True,
                **kwargs
            )

            #캐시가 있으면 새로운 로짓과 합침
            if use_kv_cache:
                kv_cache = new_kv_cache
                logits = safe_cat(logits, new_logits, dim = -2)
            else:
                logits = new_logits

            #repeat은 특정 차원으로 확장, logits.shape[-1]은 
            last_logit_indices_expanded = repeat(last_logit_indices, 'b -> b 1 c', b = batch, c = logits.shape[-1])
            last_logits = logits.gather(1, last_logit_indices_expanded)
            last_logits = rearrange(last_logits, 'b 1 c -> b c')

            filtered_logits = top_k(last_logits, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

            sampled = rearrange(sampled, 'b -> b 1')
            sample_semantic_ids = torch.cat((sample_semantic_ids, sampled), dim = -1)

            if all_rows_have_eos_id(sample_semantic_ids, self.eos_id):
                break

            last_logit_indices += 1

        sample_semantic_ids = mask_out_after_eos_id(sample_semantic_ids, self.eos_id, keep_eos = False)

        return sample_semantic_ids

    def forward(
        self,
        *,
        semantic_token_ids = None,
        raw_wave = None,
        text = None,
        text_embeds = None,
        return_loss = False,
        **kwargs
    ):
        assert exists(raw_wave) or exists(semantic_token_ids), 'either raw waveform (raw_wave) is given or semantic token ids are given (semantic_token_ids)'

        if exists(self.audio_conditioner):
            assert exists(raw_wave)
            assert not exists(text) and not exists(text_embeds)
            text_embeds = self.audio_conditioner(wavs = raw_wave, namespace = 'semantic')

        if not exists(semantic_token_ids):
            assert exists(self.wav2vec), 'VQWav2Vec must be be provided if given raw wave for training'
            semantic_token_ids = self.wav2vec(raw_wave, flatten = False)

        semantic_token_ids = rearrange(semantic_token_ids, 'b ... -> b (...)')

        if self.training:
            semantic_token_ids = append_eos_id(semantic_token_ids, self.transformer.eos_id)

        if self.unique_consecutive:
            semantic_token_ids = batch_unique_consecutive(semantic_token_ids, pad_value = self.pad_id)

        input_ids = semantic_token_ids
        if return_loss:
            input_ids = semantic_token_ids[:, :-1]

        self_attn_mask = None
        if self.mask_prob > 0. and self.training:
            self_attn_mask = generate_mask_with_prob(input_ids.shape, self.mask_prob, input_ids.device)

        logits = self.transformer(
            ids = input_ids,
            text = text,
            text_embeds = text_embeds,
            self_attn_mask = self_attn_mask,
            **kwargs
        )

        if not return_loss:
            return logits

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            semantic_token_ids,
            ignore_index = self.pad_id
        )

        return loss
