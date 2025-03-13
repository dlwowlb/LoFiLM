

class AudioLM(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        wav2vec: FairseqVQWav2Vec | HubertWithKmeans | None, 
        codec: SoundStream | EncodecWrapper,
        semantic_transformer: SemanticTransformer,
        coarse_transformer: CoarseTransformer,
        fine_transformer: FineTransformer,
        audio_conditioner: AudioConditionerBase | None = None,
        unique_consecutive = True
    ):
        super().__init__()

        self.audio_conditioner = audio_conditioner

        assert semantic_transformer.num_semantic_tokens == coarse_transformer.num_semantic_tokens
        assert coarse_transformer.codebook_size == fine_transformer.codebook_size
        assert coarse_transformer.num_coarse_quantizers == fine_transformer.num_coarse_quantizers
        assert (fine_transformer.num_coarse_quantizers + fine_transformer.num_fine_quantizers) == codec.num_quantizers

        self.semantic_has_condition = semantic_transformer.has_condition
        self.coarse_has_condition = coarse_transformer.has_condition
        self.fine_has_condition = fine_transformer.has_condition
        self.needs_text = any([self.semantic_has_condition, self.coarse_has_condition, self.fine_has_condition])

        self.semantic = SemanticTransformerWrapper(
            wav2vec = wav2vec,
            transformer = semantic_transformer,
            audio_conditioner = audio_conditioner,
            unique_consecutive = unique_consecutive
        )

        self.coarse = CoarseTransformerWrapper(
            wav2vec = wav2vec,
            codec = codec,
            transformer = coarse_transformer,
            audio_conditioner = audio_conditioner,
            unique_consecutive = unique_consecutive
        )

        self.fine = FineTransformerWrapper(
            codec= codec,
            transformer = fine_transformer,
            audio_conditioner = audio_conditioner
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.inference_mode()
    def forward(
        self,
        *,
        batch_size = 1,
        text: list[str] | None = None,
        text_embeds: Tensor | None = None,
        prime_wave = None,
        prime_wave_input_sample_hz = None,
        prime_wave_path = None,
        max_length = 2048,
        return_coarse_generated_wave = False,
        mask_out_generated_fine_tokens = False
    ):
        assert not (self.needs_text and (not exists(text) and not exists(text_embeds))), 'text needs to be passed in if one of the transformer requires conditioning'

        if self.needs_text:
            if exists(text):
                text_embeds = self.semantic.embed_text(text)

        assert not (exists(prime_wave) and exists(prime_wave_path)), 'prompt audio must be given as either `prime_wave: Tensor` or `prime_wave_path: str`'

        if exists(prime_wave):
            assert exists(prime_wave_input_sample_hz), 'the input sample frequency for the prompt audio must be given as `prime_wave_input_sample_hz: int`'
            prime_wave = prime_wave.to(self.device)
        elif exists(prime_wave_path):
            prime_wave_path = Path(prime_wave_path)
            assert exists(prime_wave_path), f'file does not exist at {str(prime_wave_path)}'

            prime_wave, prime_wave_input_sample_hz = torchaudio.load(str(prime_wave_path))
            prime_wave = prime_wave.to(self.device)

        semantic_token_ids = self.semantic.generate(
            text_embeds = text_embeds if self.semantic_has_condition else None,
            batch_size = batch_size,
            prime_wave = prime_wave,
            prime_wave_input_sample_hz = prime_wave_input_sample_hz,
            max_length = max_length
        )


        coarse_token_ids_or_recon_wave = self.coarse.generate(
            text_embeds = text_embeds if self.coarse_has_condition else None,
            semantic_token_ids = semantic_token_ids,
            prime_wave = prime_wave,
            prime_wave_input_sample_hz = prime_wave_input_sample_hz,
            reconstruct_wave = return_coarse_generated_wave
        )

        if return_coarse_generated_wave:
            return coarse_token_ids_or_recon_wave

        generated_wave = self.fine.generate(
            text_embeds = text_embeds if self.fine_has_condition else None,
            coarse_token_ids = coarse_token_ids_or_recon_wave,
            prime_wave = prime_wave,
            prime_wave_input_sample_hz = prime_wave_input_sample_hz,
            reconstruct_wave = True,
            mask_out_generated_fine_tokens = mask_out_generated_fine_tokens
        )

        return generated_wave