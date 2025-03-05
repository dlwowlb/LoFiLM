from pathlib import Path

import torch
from torch import nn
from einops import rearrange

import fairseq

from torchaudio.functional import resample


import logging
logging.root.setLevel(logging.ERROR)

def round_down_nearest_multiple(num, divisor):
    return num // divisor * divisor

def curtail_to_multiple(t, mult, from_left = False):
    data_len = t.shape[-1]
    rounded_seq_len = round_down_nearest_multiple(data_len, mult)
    seq_slice = slice(None, rounded_seq_len) if not from_left else slice(-rounded_seq_len, None)
    return t[..., seq_slice]

def exists(val):
    return val is not None

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
        wav2vec: FairseqVQWav2Vec | HubertWithKmeans | None = None,
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

    @eval_decorator
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
            ids = self.wav2vec(
                prime_wave,
                flatten = False,
                input_sample_hz = prime_wave_input_sample_hz
            )
        elif exists(prime_ids):
            ids = prime_ids
        else:
            ids = torch.empty((batch_size, 0), dtype = torch.long, device = device)

        if self.unique_consecutive:
            ids = batch_unique_consecutive(ids, pad_value = self.pad_id)

        # derive joint audio-text embeddings if needed

        if exists(self.audio_conditioner) and exists(prime_wave):
            assert not exists(text) and not exists(text_embeds)
            text_embeds = self.audio_conditioner(wavs = prime_wave, namespace = 'semantic')

        # derive text embeddings if needed

        has_text = exists(text) or exists(text_embeds)
        assert not (self.transformer.has_condition ^ has_text)

        if not exists(text_embeds) and exists(text):
            with torch.inference_mode():
                text_embeds = self.transformer.embed_text(text, output_device = device)

        # start length and get running id output

        batch = ids.shape[0]
        start_length = ids.shape[-1]
        sample_semantic_ids = ids.clone()

        last_logit_indices = (ids != self.pad_id).sum(dim = -1).long()

        # kv cache

        kv_cache = None
        logits = None

        # sample from transformer

        for ind in tqdm(range(start_length, max_length), desc = 'generating semantic'):

            new_logits, new_kv_cache = self.transformer.forward_with_cond_scale(
                ids = sample_semantic_ids,
                text_embeds = text_embeds,
                cond_scale = cond_scale,
                kv_cache = kv_cache,
                return_kv_cache = True,
                **kwargs
            )

            if use_kv_cache:
                kv_cache = new_kv_cache
                logits = safe_cat(logits, new_logits, dim = -2)
            else:
                logits = new_logits

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
