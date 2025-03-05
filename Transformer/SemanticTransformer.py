from utils import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME, default, prob_mask_like
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange


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
        kv_cache = iter(default(kv_cache, []))
        new_kv_caches = []

        logits, new_kv_cache = self.forward(*args, cond_drop_prob = 0., kv_cache = next(kv_cache, None), return_kv_cache = True, **kwargs)
        new_kv_caches.append(new_kv_cache)

        if cond_scale == 1 or not self.has_condition:
            if not return_kv_cache:
                return logits

            return logits, torch.stack(new_kv_caches)

        null_logits, null_new_kv_cache = self.forward(*args, cond_drop_prob = 1., kv_cache = next(kv_cache, None), return_kv_cache = True, **kwargs)
        new_kv_caches.append(null_new_kv_cache)

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




        if return_loss:
            labels, ids = ids.clone(), ids[:, :-1]

        tokens = get_embeds(self.semantic_embedding, ids)

        start_tokens = repeat(self.start_token, 'd -> b 1 d', b = ids.shape[0])

        tokens = torch.cat((start_tokens, tokens), dim = 1)

        if exists(self_attn_mask):
            self_attn_mask = F.pad(self_attn_mask, (1, 0), value = True)

        tokens, kv_cache = self.transformer(tokens, context = text_embeds, self_attn_mask = self_attn_mask, context_mask = text_mask, kv_cache = kv_cache, return_kv_cache = True)
        logits = self.to_logits(tokens)

        if not return_kv_cache:
            return logits

        return logits, kv_cache
