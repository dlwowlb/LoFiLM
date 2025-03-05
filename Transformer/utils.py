import torch
import transformers

from transformers import T5Tokenizer, T5EncoderModel, T5Config
from torch import nn, einsum, Tensor
from einops import rearrange, repeat, reduce

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


def exists(val):
    return val is not None

@beartype
def get_embeds(
    embeddings: nn.Embedding,
    codes: torch.Tensor,
    pad_id = -1,
    return_mask = False,
    mask_pad_pos_to = 0
):
    pad_mask = codes == pad_id
    codes_without_pad = codes.masked_fill(pad_mask, 0) # 패딩 위치에 0 넣기
    embeds = embeddings(codes_without_pad)

    #mask_pad_pos_to가 값이 있으면 패딩 위치에 0대신 mask_pad_pos_to 값으로 채움
    if exists(mask_pad_pos_to):
        embeds = embeds.masked_fill(rearrange(pad_mask, '... -> ... 1'), mask_pad_pos_to)

    #필요하다면 마스크 리턴
    if return_mask:
        return embeds, ~pad_mask

    return embeds

#확률에 따라 무작위 True/False 마스크 생성
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        dim_context = None,
        cross_attend = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        grad_shrink_alpha = 0.1,
        cond_as_self_attn_prefix = False,
        rel_pos_bias = True,
        flash_attn = False,
        add_value_residual = True,
        num_residual_streams = 4,
        **kwargs
    ):
        super().__init__()
        rel_pos_bias = rel_pos_bias and not flash_attn

        assert not (cross_attend and cond_as_self_attn_prefix)

        self.dim_context = default(dim_context, dim)

        self.cond_as_self_attn_prefix = cond_as_self_attn_prefix

        self.grad_shrink = partial(grad_shrink, alpha = grad_shrink_alpha)

        self.layers = nn.ModuleList([])

        self.rel_pos_bias = RelativePositionBias(dim = dim // 2, heads = heads) if rel_pos_bias else None

        # hyper connections

        init_hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        # layers

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                init_hyper_conn(dim = dim, branch = Attention(dim = dim, heads = heads, dropout = attn_dropout, flash = flash_attn, causal = True, **kwargs)),
                init_hyper_conn(dim = dim, branch = Attention(dim = dim, heads = heads, dropout = attn_dropout, dim_context = dim_context, flash = flash_attn, num_null_kv = 1, norm_context = True, **kwargs)) if cross_attend else None,
                init_hyper_conn(dim = dim, branch = FeedForward(dim = dim, dropout = ff_dropout))
            ]))

        self.norm = LayerNorm(dim)

        self.add_value_residual = add_value_residual

    def forward(
        self,
        x,
        self_attn_mask = None,
        context = None,
        context_mask = None,
        attn_bias = None,
        return_kv_cache = False,
        kv_cache = None
    ):
        assert not (self.cond_as_self_attn_prefix and not exists(context))
        assert not (exists(context) and context.shape[-1] != self.dim_context), f'you had specified a conditioning dimension of {self.dim_context}, yet what was received by the transformer has dimension of {context.shape[-1]}'

        n, device = x.shape[1], x.device

        # from cogview paper, adopted by GLM 130B LLM, decreases likelihood of attention net instability

        x = self.grad_shrink(x)

        # turn off kv cache if using conditioning as self attention (as in valle), for now

        if self.cond_as_self_attn_prefix:
            kv_cache = None

        # handle kv cache

        new_kv_cache = []

        if exists(kv_cache):
            cache_len = kv_cache.shape[-2]
            kv_cache = iter(kv_cache)
        else:
            cache_len = 0
            kv_cache = iter([])

        x = x[:, cache_len:]

        # relative positional bias

        if exists(attn_bias):
            rel_pos_bias = attn_bias
        else:
            rel_pos_bias = maybe(self.rel_pos_bias)(n, n)

        if exists(rel_pos_bias):
            rel_pos_bias = rel_pos_bias[..., cache_len:, :]

        # self attention kwargs

        self_attn_kwargs = dict()
        if self.cond_as_self_attn_prefix:
            self_attn_kwargs = dict(
                prefix_context = context,
                prefix_context_mask = context_mask
            )

        # value residuals

        self_attn_value_residual = None
        cross_attn_value_residual = None

        # expand residual streams

        x = self.expand_streams(x)

        # transformer layers

        for attn, cross_attn, ff in self.layers:

            residual = x

            x, (layer_kv_cache, values) = attn(x, attn_bias = rel_pos_bias, mask = self_attn_mask, kv_cache = next(kv_cache, None), return_kv_cache = True, return_values = True, value_residual = self_attn_value_residual, **self_attn_kwargs)

            if self.add_value_residual:
                self_attn_value_residual = default(self_attn_value_residual, values)

            new_kv_cache.append(layer_kv_cache)

            if exists(cross_attn):
                assert exists(context)

                x, values = cross_attn(x, context = context, mask = context_mask, return_values = True, value_residual = cross_attn_value_residual)

                if self.add_value_residual:
                    cross_attn_value_residual = default(cross_attn_value_residual, values)

            x = ff(x)

        # reduce residual streams

        x = self.reduce_streams(x)

        # final norm

        x = self.norm(x)

        if not return_kv_cache:
            return x

        return x, torch.stack(new_kv_cache)
