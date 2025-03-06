# LoFiLM

Reimplementation and Finetunning using https://github.com/lucidrains/audiolm-pytorch.git

Todo

- [x]  AudioTransformer 모듈 만들기
- [x]  TextTransformer 모듈 만들기
- [x]  Mulan 모듈로 Audio, Text를 공통된 latent 차원으로 변환
- [x]  Text, Audio 두 modality의 임베딩 간의 contrastive loss 모듈 만들기
- [x]  Mulan의 출력을 RVQ 코드북으로 quantize
- [ ]  Audiolm을 사용한 Semantic, Coarse, Fine 조건 임베딩 토큰 생성
    - [ ] Semantic Transformer
    - [ ] Coarse Transformer
    - [ ] Fine Trnasformer
- [ ]  토큰 생성 후 soundstream decoder로 여러 음악 샘플 생성 및 MuLaN으로 selecting
- [ ]  Lo-Fi 스타일만 생성하도록 fine-tunning, AWS 이용
- [ ]  허깅페이스 space에 올리기


------------------------------------------------------------------------------------------
- [ ]  분산학습 프로세스 다시 이해
- [ ]  ResidualVQ 모듈 다시 이해(이미 정의되어 있음)
- [ ]  추상 베이스 클래스의 역할 이해
- [ ]  X_Clip (BPE 없이) 토크나이저
- [ ]  Semantic, Coarse, Fine Transformer의 차이점
