from beartype.typing import List, Optional, Tuple
from beartype import beartype
from x_clip.tokenizer import tokenizer


class textTransformer:
    @beartype
    def __init__(
        self,
        dim,
        depth,
        num_tokens = tokenizer.vocab_size,
        max_seq_len = 256,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        pad_id = 0
        ):

        super().__init__()
        self.dim = dim
        
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.pad_id = pad_id

    @beartype
    def forward(
            self,
            x = None,
            raw_texts: Optional[List[str]] = None,
            mask = None,
            return_all_layers = False
    ):
        assert exists(x) ^ exists(raw_texts) # raw text 혹은 x가 존재해야 함

        if exists(raw_texts):
            #openAI tokenizer를 통해 raw text를 tokenize
            x = tokenizer.tokenize(raw_texts).to(self.device)

        #패딩 토큰(pad_id)이 아니면 True, 패딩 토큰이면 False 불리언 마스크 생성
        if not exists(mask):
            mask = x != self.pad_id

        b, n, device = *x.shape, x.device
        
        #토큰 임베딩
        x = self.token_emb(x)

        #포지션 임베딩
        