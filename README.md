# LoFiLM

Reimplementation and Finetunning using https://github.com/lucidrains/audiolm-pytorch.git

Todo

- [x]  audiospectogramtransformer 모듈 만들기
- [ ]  texttransformer 모듈 만들기
- [ ]  Mulan 모듈 만들기
- [ ]  더미 데이터로 위 두 개 모듈 사용하여 임베딩 추출, Mulan을 통해 공통된 latent 차원으로 변환
- [ ]  Mulan 모델은 text, audio 두 modality의 임베딩 간의 유사성을 contrastive loss를 통해 학습
- [ ]  Mulan의 출력을 RVQ 코드북으로 quantize
- [ ]  audiolm의 부은 ‘solve’ music을 위해 필수적이다. 이를 위해 Semantic, Coarse, Fine 조건 임베딩 생성

    
- [ ]  3개의 조건임베딩으로 3개의 토큰
- [ ]  토큰 생성 후 soundstream decoder로 여러 음악 샘플 생성 및 MuLaN으로 selecting
- [ ]  Lo-Fi 스타일만 생성하도록 fine-tunning, AWS 이용
- [ ]  허깅페이스 space에 올리기
