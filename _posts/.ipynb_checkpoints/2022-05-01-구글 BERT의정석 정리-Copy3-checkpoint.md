---
layout: post
title: 구글 BERT의 정석 정리 3장
categories: [NLP]
tags: [NLP]
description: 허깅 페이스의 Transformer
---

## 학습 내용

- 허깅 페이스의 Transformer 라이브러리를 활용하여 사전 임베딩 된 BERT모델에서 임베딩 추출 방법 익히기
- BERT의 모든 인코더 레이어에서 임베딩 추출 방법 익힘
- 텍스트 분류에서 파인튜닝
- 자연어 추론 (Natural Language Inference) 에서 파인튜닝

3.1 사전 학습된 bert 모델

사용방식은 아래 2가지

1. 임베딩을 추출하는 특징 추출기
2. 문제정의에 맞게 (텍스트분류/질문-응답 등) 파인 튜닝

3.2 임베딩 추출 방법

1. 각문장의 처음과 끝에 [CLS] , [SEP] 추가
2. maxlen에 맞게 문장 길이가 짧으면 끝에 [PAD] 추가
3. PAD를 제외한 토큰에 attention mask 설정 (PAD면 0 아니면 1)
4. 각 토큰에 고유 토큰 ID 매핑

일반적으로 전체 문장의 표현은 [CLS] 토큰의 임베딩을 사용함.

하지만 모든 토큰의 표현을 평균하거나 풀링 할 수도 있음

3.4 파인 튜닝 방법

3.4.1 텍스트 분류

[CLS]임베딩만 사용하여 분류기에 학습

가중치 업데이트는

1. 분류 계층과 함께 사전 학습된 BERT 모델의 가중치를 업데이트 
2. 분류계층만 업데이트 (사전 BERT 모델의 가중치는 업데이트 되지 않으므로 특징 추출기 역할함)

할 수 있음 

3.4.2 전처리 방법

총 3개의 벡터가 필요함

1. token id → 각 토큰의 매핑된 고유 id
2. token type id → 문장구분, segment id임
3. attention mask → PAD면 0, 아니면 1

3.4.2 자연어 추론

전제/ 가설/ LABEL 필요

전제와 가설을 INPUT으로 두고 임베딩 된 결과를 NN의 INPUT, LABEL을 OUTPUT으로 두고

확률을 계산함

3.4.3 질문-응답

질문후 단락에서 응답에 해당하는 내용을 찾아서 index 매핑

응답의 값의 시작과 끝의 index를 소프트 맥스 함수를 통해 찾음

*관련 코드*

[https://colab.research.google.com/drive/1xewNLldlqA-s5cPaW9SvUcOCb_k754G8?usp=sharing](https://colab.research.google.com/drive/1xewNLldlqA-s5cPaW9SvUcOCb_k754G8?usp=sharing)