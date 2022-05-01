---
layout: post
title: 구글 BERT의 정석 정리 4장
categories: [NLP]
tags: [NLP]
description: BERT 파생모델
---
BERT의 4개 파생모델에 대해 살펴봄

- ALBERT ( Lite version of BERT)
- RoBERTa (Robustly Optimized BERT pre-training Approach)
- ELECCTRA ( Efficiently Learning and Encoder that Classifier Token Replacements Accurately)
- SpanBERT  (span 예측)

### 4.1 ALBERT

BERT는 학습 파라미터가 많아서 시간이 오래 걸린다는 단점이 있음, 아래 2가지 방법으로 학습시간과 추로 시간을 줄임

- 크로스 레이어 변수 공유
- 팩토라이즈 임베딩 레이어 변수화

4.1.1 크로스 레이어 변수 공유

BERT에는 12개의 인코더가 존재하는데, 첫번째 인코더의 값을 나머지 인코더들에 공유하는 방식

- All-shared : 첫번째 인코더 하위 레이어의 모든 변수를 나머지 인코더와 공유
- Shared feedforward network : 첫번째 인코더의 feedforward network 변수만 공유
- SHared attention : 첫번째 인코더의 멀티헤드어텐션 변수만 공유

4.1.2 팩토라이즈 임베딩 레이어 변수화

parameter reduction 방법. BERT의 사전크기를 30,000 , 은닉 레이어 크기를 768 이라고 하면

은닉레이어와 워드피스 임베딩 크기는 (30,000*768)임

하지만 행렬분해를 통해 원핫벡터의 BERT 사전크기를 128의 워드피스 임베딩에 투영시킨 후, 

이것을 다시 은닉레이어로 투영시킴

이때 워드피스 임베딩 크기는 (30,000*128 )  크기로 투영 후 최종적으로  (128* 768)이 된다.

4.1.3 ALBERT 모델 학습 

MLM은 사용 , NSP 대신 문장 순서 예측 (SOP) 사용

NSP는 문서내 문장과 다른 문서의 문장을 비교하는데, 

Sentence Order Prediction (SOP)

NSP는 문장이 다음 문장인지 아닌지 예측하는 형태였지만 SOP는 문장 순서가 바뀌었는지 예측함. 즉 2개의 문장이 연속되면 positive, 아니면 negative임

- positive 예

그녀는 파스타를 요리했다.

파스타는 맛있었다.

- negative 예

파스타는 맛있었다.

그녀는 파스타를 요리했다.

### 4.2 RoBERTa

BERT의 사전 학습방식을 변경함

- MLM에서 정적 마스킹 대신 동적 마스킹
- NSP를 태스크 제거
- 배치 크기 증가
- 토크나이저로 워드피스 대신 BBPE 사용

4.2.1 정적마스크 대신 동적마스크

정적 마스크

tokens = [ [CLS] , we , arrived, at, the, airport , in ,the, time, [SEP]]

→ tokens = [ [CLS] , we , [MASK], at, the, airport , in ,[MASK], time, [SEP]]

1 Epoch에서 동일한 마스킹을 예측하도록 학습이 이루어 짐

동적 마스크

문장을 10개를 복사 후 각 10개의 문장별로 mask를 다르게 함

1. tokens = [ [CLS] , we , arrived, at, the, [MASK], in ,the, time, [SEP]]
2. tokens =  [ [CLS] , we , [MASK], [MASK], the, airport , in ,the, time, [SEP]]

...

1. tokens = [ [CLS] , [MASK], arrived, at, the, airport , in ,[MASK], time, [SEP]]

그 후 40번의 에폭이 있다라고 하면 

- 1번문장은 1,11,21,31 에폭때 사용됨
- 2번문장은 2,12,22,32 에폭때 사용됨

4.2.3. NSP 태스크 제거

4가지 실험을 진행

- SEGMENT PAIR + NSP : 기존 버트, 입력값을 동일 문서에서 추출
- SENTENCE PAIR + NSP : 입력값을 여러 문서에서 추출
- FULL SENTENCE : 입력값을 여러 문서에서 추출
- DOC SENTENCE : 입력값을 하나의 문서에서 추출

NSP 를 하나 안하나 비슷하더라..

DOC SENTENCE는 문서에 따라 배치 크기가 달라지므로 FULL SETENCE 방식을 사영

4.2.4 그외

- 더많은 데이터 (40G → 160G)
- 더큰 배치 (256배치, 100만 epoch → 8000배치 , 50만 epoch)
- 워드피스 → BBPE

### 4.3 ELECTRA

MLM 대신, 교체한 토큰 탐지 사용(replaced token detection). MASK 대신 토큰이 실제 토큰인지 교체된 토큰인지 판별하는 형태로 학습

- 사전 학습에 사용되는 MASK와 파인 튜닝 때 사용되는 MASK간 불일치 발생
- 토큰을 다른 토큰으로 대체 후 실제 토큰인지 대체된 토큰인지 분류 하도록 모델 학습
- 교체된 토큰만을 탐지하는 태스크만 사전학습을 진행

4.3.1 교체한 토큰 탐지 이해

tokens = [The , chef, cooked , the, meal]

→ replace tokens = [a, chef, ate, the, meal]

a      → replaced

chef → original

ate   → replaced

the   → original

meal  → original

위와 같이 분류하는 판별자를 학습함.

그렇다면 어떻게 replaced 된 token을 제공하는 가?

이를 위해서 MLM을 사용함.

원래 bert에선 아래와 같이 학습됨

tokens = [The , chef, cooked , the, meal]

→ making tokens = [[MASK], chef, [MASK], the, meal]

이때 [MASK] 토큰을 예측할 때 토큰에 대한 값이 확률분포로 제공됨

- 예측된 값이 [[a], chef, [ate], the, meal] 로 나오고 해당 값을 사용함

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9cdfbc80-96b3-4fd1-abbd-e02565beef6a/Untitled.png)

즉, token이 replace 된건지 original 인지 생성하기 위해 MLM을 사용하고 이를 분류하는 판별자를 ELECTRA라고 함.

4.3.2 생성자와 판별자 이해

생성자는 MLM 태스크를 수행

- 무작위로 일부 토큰을 [MASK] 하고 , 해당 토큰의 표현을 소프트맥스 함수를 통해 확률값 출력

판별자는 생성자에 의해 만들어진 토큰인지, 원래 토큰인지 구별함

- 생성자에 의해 mask 표현 중 높은 확률값을 가지는 token을 input으로 가지고 원래 토큰인지 아닌지를 구별하는 분류기를 학습

위 방식을 통해 아래와 같은 장점

- 기존에는 전체 토큰의 15%정도만 마스킹하여 사용하지만
- 주어진 토큰의 원본 여부를 판별하는 방식이어서, 전체 토큰을 대상으로 학습 이뤄짐

 

4.3.3 모델 학습 방식

min(생성자 LOSS + a* 판별자 LOSS)

- 생성자 LOSS :

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5881d127-48c6-4ab9-9137-ea987ecf6110/Untitled.png)

- 판별자 LOSS :

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7f83992f-3a7f-431f-b147-69c398a9aa96/Untitled.png)

GAN과 비슷한듯 다른데, 

- GAN은 adversial network 이지만, 위 방식은 생성자에서 샘플링이 발생해서 역전파 학습이 힘들고, maximum likelihood 방식으로 학습
- generator에서 동일한 token을 생성하면 positive sample로 간주 (GAN은 negative 하게 간주)
- generator에서 noise 함수 추가 하지 않음 (GAN은 추가함)

그외 )

- 효율적 학습을 위해 생성자 판별자 가중치를 공유

### 4.4 SpanBERT

텍스트 범위를 예측하는 질문-응답 태스크에 주로 이용됨

4.4.1 아키텍쳐

tokens = [you , are , expected, to , know, the, laws, of ,your, country]

→  [you , are , expected, to , know, [MASK1], [MASK2], [MASK3],[MASK4], country]

즉 토큰의 위치를 무작위로 mask 하지않고, 토큰의 범위를 mask 함

- 기존 MLM 에 새로운 목적 함수 SBO(span boundary objective) 사용

SBO ?

[MASK]를 예측하기 위해 span 시작전 , 종료 후 token 만을 사용함

예를들어 [MASK2]를 예측하기 위해 [know] 와 [country] 만을 사용함

하지만 span 내 [MASK1], [MASK2]... 들은 어떻게 구분될까?

즉 [MASK1] [MASK2] [MASK3] [MASK4] 를 예측하기 위해 input이 [know]와 [country]로 동일한 문제가 있음

⇒이를 위해 위치 임베딩(P1, P2, P3,P4) 을사용함

4.4.2 spanbert 탐색

spanbert의 손실함수는

- 기존 MLM 에서 얻어진 각 토큰의 임베딩 값
- SBO에서 계산된, 위치 임베딩이 포함된 정보를 사용

SBO에선

이때 기본적으로 2개의 피드포워드 네트워크에 GelU 함수로 구성됨

h0 = [span 시작전 , span 종료후 , 위치벡터]

h1 = [첫번째 gelu]

z1 = [두번째 gelu]

기존 MLM 벡터 + z1 사용
