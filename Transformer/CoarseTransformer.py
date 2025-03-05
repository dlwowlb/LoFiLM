from torch import nn
from audiolm_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME


class CoarseTransformer(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        codebook_size,
        num_coarse_quantizers,
        dim,
        depth,
        num_semantic_tokens,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        t5_name = DEFAULT_T5_NAME,
        has_condition = False,
        cond_dim = None,
        audio_text_condition = False,
        cond_as_self_attn_prefix = False,
        cond_drop_prob = 0.5,
        grad_shrink_alpha = 0.1,
        project_semantic_logits = True,
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

        self.semantic_start_token = nn.Parameter(torch.randn(dim))
        self.coarse_start_token = nn.Parameter(torch.randn(dim))

        self.semantic_eos_id = num_semantic_tokens
        self.semantic_embedding = nn.Embedding(num_semantic_tokens + 1, dim)

        self.coarse_eos_id = codebook_size
        codebook_size_with_eos = codebook_size + 1

        self.coarse_embedding = nn.Embedding(num_coarse_quantizers * codebook_size_with_eos, dim)
        self.coarse_quantize_embedding = nn.Embedding(num_coarse_quantizers, dim)

        text_dim = default(cond_dim, get_encoded_dim(t5_name))
        self.proj_text_embed = nn.Linear(text_dim, dim, bias = False) if text_dim != dim else nn.Identity()

        self.cross_attn_bias = nn.Parameter(torch.zeros(heads, 1, 1)) if rel_pos_bias else None

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

        self.codebook_size = codebook_size
        self.num_coarse_quantizers = num_coarse_quantizers

        self.to_semantic_logits = nn.Linear(dim, num_semantic_tokens + 1) if project_semantic_logits else None
        self.coarse_logit_weights = nn.Parameter(torch.randn(num_coarse_quantizers, codebook_size_with_eos, dim))

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
        return_kv_cache = False,
        kv_cache = None,
        embed_cache = None,
        **kwargs
    ):
        iter_kv_cache = iter(default(kv_cache, []))
        iter_embed_cache = iter(default(embed_cache, []))
        new_kv_caches = []
        new_embed_caches = []

        (semantic_logits, coarse_logits), (new_kv_cache, new_embed_cache) = self.forward(*args, cond_drop_prob = 0., return_cache = True, kv_cache = next(iter_kv_cache, None), embed_cache = next(iter_embed_cache, None), **kwargs)
        new_kv_caches.append(new_kv_cache)
        new_embed_caches.append(new_embed_cache)

        if cond_scale == 1 or not self.has_condition:
            if not return_kv_cache:
                return semantic_logits, coarse_logits

            return (semantic_logits, coarse_logits), (torch.stack(new_kv_caches), torch.stack(new_embed_caches))

        (null_semantic_logits, null_coarse_logits), (null_new_kv_cache, null_new_embed_cache) = self.forward(*args, cond_drop_prob = 1., return_cache = True, kv_cache = next(iter_kv_cache, None), embed_cache = next(iter_embed_cache, None), **kwargs)
        new_kv_caches.append(null_new_kv_cache)
        new_embed_caches.append(null_new_embed_cache)

        scaled_semantic_logits = None
        if exists(null_semantic_logits):
            scaled_semantic_logits = null_semantic_logits + (semantic_logits - null_semantic_logits) * cond_scale

        scaled_coarse_logits = null_coarse_logits + (coarse_logits - null_coarse_logits) * cond_scale

        if not return_kv_cache:
            return scaled_semantic_logits, scaled_coarse_logits

        return (scaled_semantic_logits, scaled_coarse_logits), (torch.stack(new_kv_caches), torch.stack(new_embed_caches))

    @beartype
    def forward(
        self,
        *,
        semantic_token_ids,
        coarse_token_ids,
        self_attn_mask = None,
        text: list[str] | None = None,
        text_embeds = None,
        cond_drop_prob = None,
        return_only_coarse_logits = False,
        return_cache = False,
        kv_cache = None,
        embed_cache = None
    ):
        #semantic_tokens은 semantic.generate에서 생성됨
        #semantic은 semanticTransformerWrapper에서 생성됨

        #semantic token의 코드북에서 인덱스 가져옴옴
        b, device = semantic_token_ids.shape[0], semantic_token_ids.device 
        
        #arange 함수 쓸 때 device로 옮김김
        arange = partial(torch.arange, device = device)

        #text가 존재하거나 text_embeds가 존재해야 함
        has_text = exists(text) or exists(text_embeds)
        assert not (self.has_condition ^ has_text)

        #text_embeds가 존재하지 않고 text가 존재하면
        if not exists(text_embeds) and exists(text):
            #text를 토큰 임베딩으로 변환
            with torch.inference_mode(): #torch.no_grad보다 더 최적화
                text_embeds = self.embed_text(text, output_device = device)

        text_mask = None
        if exists(text_embeds):
            text_mask = torch.any(text_embeds != 0, dim = -1)

            text_embeds = self.proj_text_embed(text_embeds)

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        if exists(text_mask) and cond_drop_prob > 0:
            keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        coarse_token_ids, semantic_token_ids = map(lambda t: rearrange(t, 'b ... -> b (...)'), (coarse_token_ids, semantic_token_ids))

        offsets = self.codebook_size * arange(self.num_coarse_quantizers)
        offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(coarse_token_ids.shape[-1], self.num_coarse_quantizers))
        offsets = offsets[:, :coarse_token_ids.shape[-1]]
        coarse_token_ids = coarse_token_ids + offsets

        semantic_tokens = get_embeds(self.semantic_embedding, semantic_token_ids)
        coarse_tokens = self.coarse_embedding(coarse_token_ids)

        coarse_quantize_tokens = repeat(self.coarse_quantize_embedding.weight, 'q d -> (n q) d', n = ceil_div(coarse_token_ids.shape[-1], self.num_coarse_quantizers))
        coarse_quantize_tokens = coarse_quantize_tokens[:coarse_token_ids.shape[-1], ...]
        coarse_tokens = coarse_tokens + coarse_quantize_tokens

        semantic_seq_len = semantic_tokens.shape[1]

        semantic_start_tokens = repeat(self.semantic_start_token, 'd -> b 1 d', b = b)
        coarse_start_tokens = repeat(self.coarse_start_token, 'd -> b 1 d', b = b)

        tokens = torch.cat((
            semantic_start_tokens,
            semantic_tokens,
            coarse_start_tokens,
            coarse_tokens
        ), dim = 1)

        # engineer the attention bias so that cross attention is not dominated by relative positions

        seq_len = tokens.shape[-2]

        attn_bias = None

        if exists(self.transformer.rel_pos_bias):
            attn_bias = self.transformer.rel_pos_bias(seq_len, seq_len)

            is_semantic = arange(seq_len) < (semantic_seq_len + 1) # semantic seq len + start token
            is_cross_attn = rearrange(is_semantic, 'i -> i 1') ^ rearrange(is_semantic, 'j -> 1 j')

            attn_bias = torch.where(
                is_cross_attn,
                self.cross_attn_bias,
                attn_bias
            )

        # attend

        tokens, new_kv_cache = self.transformer(
            tokens,
            context = text_embeds,
            attn_bias = attn_bias,
            self_attn_mask = self_attn_mask,
            context_mask = text_mask,
            kv_cache = kv_cache,
            return_kv_cache = True
        )

        if exists(embed_cache):
            tokens = torch.cat((embed_cache, tokens), dim = -2)

        new_embed_cache = tokens

        # segment into semantic and coarse acoustic tokens

        pred_semantic_tokens, pred_coarse_tokens = tokens[:, :semantic_seq_len], tokens[:, (semantic_seq_len + 1):]

        # semantic logits

        semantic_logits = self.to_semantic_logits(pred_semantic_tokens) if not return_only_coarse_logits and exists(self.to_semantic_logits) else None

        # get coarse logits

        n = pred_coarse_tokens.shape[1]
        nq = round_down_nearest_multiple(n, self.num_coarse_quantizers)

        pred_coarse_tokens_groupable, pred_coarse_tokens_remainder = pred_coarse_tokens[:, :nq], pred_coarse_tokens[:, nq:]

        pred_coarse_tokens_groupable = rearrange(pred_coarse_tokens_groupable, 'b (n q) d -> b n q d', q = self.num_coarse_quantizers)

        coarse_logits_groupable = einsum('q c d, b n q d -> b n q c', self.coarse_logit_weights, pred_coarse_tokens_groupable)

        coarse_logits_groupable = rearrange(coarse_logits_groupable, 'b n q c -> b (n q) c')

        remainder_num_quantizers = pred_coarse_tokens_remainder.shape[1]

        if remainder_num_quantizers > 0:
            coarse_logits_remainder = einsum('q c d, b q d -> b q c', self.coarse_logit_weights[:remainder_num_quantizers], pred_coarse_tokens_remainder)

            coarse_logits = torch.cat((coarse_logits_groupable, coarse_logits_remainder), dim = 1)
        else:
            coarse_logits = coarse_logits_groupable

        logits = (semantic_logits, coarse_logits)

        if not return_cache:
            return logits

        return logits, (new_kv_cache, new_embed_cache)