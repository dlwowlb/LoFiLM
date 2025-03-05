import torch
import transformers
from transformers import T5Tokenizer, T5EncoderModel, T5Config

from beartype import beartype
from beartype.typing import Union, List

# less warning messages since only using encoder

transformers.logging.set_verbosity_error()

# helper functions

def exists(val):
    return val is not None

# config

MAX_LENGTH = 256

DEFAULT_T5_NAME = 'google/t5-v1_1-base'

T5_CONFIGS = {}

#T5 토크나이저
def get_tokenizer(name):
    tokenizer = T5Tokenizer.from_pretrained(name)
    return tokenizer

#T5 모델
def get_model(name):
    model = T5EncoderModel.from_pretrained(name)
    return model

#전역 캐시 저장(재사용)
def get_model_and_tokenizer(name):
    global T5_CONFIGS

    if name not in T5_CONFIGS:
        T5_CONFIGS[name] = dict()

    if "model" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["model"] = get_model(name)

    if "tokenizer" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["tokenizer"] = get_tokenizer(name)

    return T5_CONFIGS[name]['model'], T5_CONFIGS[name]['tokenizer']

def get_encoded_dim(name):
    if name not in T5_CONFIGS:
        config = T5Config.from_pretrained(name)
        T5_CONFIGS[name] = dict(config = config)

    elif "config" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["config"]

    elif "model" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["model"].config

    else:
        raise ValueError(f'unknown t5 name {name}')

    return config.d_model

# encoding text

@beartype
def t5_encode_text(
    texts: Union[str, List[str]],
    name = DEFAULT_T5_NAME,
    output_device = None
):
    
    #단일 문자면 리스트 변환환
    if isinstance(texts, str):
        texts = [texts]

    t5, tokenizer = get_model_and_tokenizer(name)

    if torch.cuda.is_available():
        t5 = t5.cuda()

    device = next(t5.parameters()).device

    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors = 'pt',
        padding = 'longest',
        max_length = MAX_LENGTH,
        truncation = True
    )
    
    #batch_encode_plus에서 input_ids와 attention_mask를 가져옴
    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    t5.eval()

    with torch.inference_mode():
        output = t5(input_ids = input_ids, attention_mask = attn_mask)
        encoded_text = output.last_hidden_state.detach()

    attn_mask = attn_mask[..., None].bool()

    if not exists(output_device):
        encoded_text = encoded_text.masked_fill(~attn_mask, 0.)
        return encoded_text

    encoded_text.to(output_device)
    attn_mask.to(output_device)

    encoded_text = encoded_text.masked_fill(~attn_mask, 0.)
    return encoded_text


#val이 존재하면 val 반환, 아니면 d 반환
def default(val, d):
    return val if exists(val) else d

#확률에 따라 무작위 True/False 마스크 생성
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob
