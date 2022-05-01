---
layout: post
title: 구글 BERT의 정석 정리 2장
categories: [NLP]
tags: [NLP]
description: BERT 란?
---
# 2장 bert이해하기

MLM (마스크 언어 모델링) 과 NSP (다음문장 예측) 이라는 2개의 main task가 존재

2.1 기본개념

word2vec은 문맥이 고려되지 않은 임베딩 이지만, bert는 문맥까지 고려함

word2vec은 python이라는 단어는 동일한 임베딩을 가짐 하지만 bert는 문장에 따라 다른 임베딩 제공

2.2 동작방식

트랜스포머의 인코더만 사용

2.3 구조

두개의 구성모델 제시

BERT-base

BERT-large

2.3.1 BERT-base

12개의 인코더, 12개의 어텐션헤드, 768개의 hidden layer 존재. 

학습 파라미터 약 1억 1천만개

2.3.2 BERT-larget

24개 인코더, 16개 어텐션헤드, 1024개의 hidden layer. 

학습파라미터 약 3억 4천만개

그 외 다양한 BERT 존재

2.4 사전 학습

대량의 데이터로 학습된 Parameter를 초기가중치로 사용함

2.4.1 입력표현

3가이 임베딩 레이어를 기반으로 입력데이터를 변환 해야함

토큰 임베딩/ 세그먼트 임베딩/ 위치 임베딩

CLS : 문장의 시작점을 알려줌

SEP : 문장이 끝점을 알려줌

입력 : [CLS] Paris is a beatiful city [SEP] I love Paris [SEP]

토큰 : E_cls + E_paris ...                                             + E_sep

세그 : E_A + E_A +                               +E_B +         + E_B

위치 : E_0 + E_1 +                                                    + E_10

워드피스 토크나이저

하위단어 토크나이징

Pretraining → pre / ##train / ##ing 로 구분됨

토큰화 할 때 단어가 어휘사전에 있는지 확인함. oov처리에 유용. 

2.4.2 사전 학습 전략

1. 마스크언어모델링 MLM
2. 다음 문장 예측 NSP

MLM

자동인코딩 언어모델. 입력 문장에서 전체 단어의 약 15%를 masking 후 해당 단어를 예측

A: Paris is a beatiful city

B : I love Paris

를 예제로 살펴보면

token = [ [CLS], Paris is a beatutiful city [SEP] I love Paris [SEP]] 임

15%를 masking. 여기선 city로 진행하면 

token = [ [CLS], Paris is a beatutiful [MASK] [SEP] I love Paris [SEP]] 

하지만 파인튜닝 할 때 [MASK] 입력이 없어서 문제가 발생. 이를 극복하기 위해 80-10-10 규칙적용

1. 15%중 80% 는 mask 사용
2. 15%중 10%는 임의 단어 사용
3. 15%중 10%는 어떤 변경도 하지 않음.

전체 단어 마스킹 (WWM)

tokens = [let , us, start, pre, ##train, ##ing , the model] 이 있다면

tokens = [ [CLS], let , us, start, pre, ##train, ##ing , the model, [SEP]]

로 변환 후

tokens = [ [CLS], [mask] , us, start, pre, [mask], ##ing , the model, [SEP]] 가 됨.

##train 의 상위 토큰들 또한 maskig 됨

따라서 

tokens = [ [CLS], [mask] , us, start, [mask], [mask], [mask] , the model, [SEP]]

가 되는데 15% 비율이 초과되므로 let에 mask가 제거됨 

tokens = [ [CLS], let , us, start, [mask], [mask], [mask] , the model, [SEP]]

다음문장 예측(NSP)

이진분류, 문장간 관계 파악 가능, 문장이 이어진 경우는 label 을 isNext, 안이어진 경우를 NotNext로 두고 , 두 클래스의 비율을 각각 50%씩 유지함.

R_[cls] 만 가지고 2개의 NSP task 수행함. 전체 문맥이포함된 내용이기도 하고, 복잡도를 줄이기 위해 하나만 사용. 

2.4.3 사전 학습 절차

MLM과 NSP 작업 동시에 사용

100만스텝 학습, batch = 256, optimizer = adam, 

lr = 1e-4 , warmup은 1만스텝 , dropout = 0.1, gelu 사용

warmup이란?

초기엔 학습률을 높이고, 학습 될수 록 학습률을 낮춰 학습.

초기 1만 스텝은 0 ~ 1e-4 까지 선형으로 증가 시키고

그 후에는 수렴에 가까워 짐에 따라 선형으로 감소 시킴

2.5 하위 단어 토큰화 알고리즘

vocabulary = [game, the, I, played, walked, enjoy] 가 있다고 하고

입력 단어가 [I enjoyed the game ]이라 하면

enjoy는 있지만 enjoyed 는 없으므로, [I , <UNK> , the game] 이 된다.

하지만 enjoy를 쓸수 있는, 더 좋은 방법은 없을까?

vocabulary 를  [game, the, I, play, walk, ed, enjoy]로 나눠 생각해보자.

그럼  [I ,enjoy, ##ed , the ,game] 으로 토큰이 구성된다. 이런식으로 하위 단어 토큰화 알고리즘에 대해 알아본다

- 바이트 쌍 인코딩 (BPE)
- 바이트 수준 바이트 쌍 인코딩
- 워드피스

2.5.1 바이트 쌍 인코딩 

위 고유 문자들을 추출하면

어휘사전  = [a, b, c, e, l, m, n ,o, s, t, u] 임

이때 문자 시퀀스 에서 st 라는 쌍이 4번 등장하여 어휘사전에 st 추가

me는 쌍 3번 등장하여 me를 어휘 사전에 추가

me 와 n 쌍이 2번 등장하여 men을 어휘 사전에 추가

최종적으로 어휘사전  = [a, b, c, e, l, m, n ,o, s, t, u, st, me, men]임

만약 bear가 들어오면

be / ar 로 나누고 be 는 있지만 ar은 없으므로

tockens = [be,a,<UNK>] 로 구분됨

2.5.2 바이트 수준 바이트 쌍 인코딩

문자 시퀀스 best 가 있으면, 바이트 수준 시퀀스로 변환

문자 시퀀스 : b e s t 

바이트 시퀀스 : 62 65 73 74

이렇게 하면 다국어 설정에 유용, oov 처리에 효과적

2.5.3 워드피스

워드피스는 bpe와 유사하지만, 한 차이가 있음 

bpe : 토큰화 후 빈도가 많은 경우를 묶음

워드피스 : 토큰화 후 가능도가 가장 큰 경우를 묶음

만약 s와 t의 가능도를 계산한다면

p(st) / p(s)p(t) 로 계산됨

만약 워드 피스로 계산된 어휘사전이 아래와 같다면

어휘사전 = {a, b, c, e, l, m, n, o, s, t, u, st, me}

예를들어 stem 을 분할하면 st / em 이 되고, st는 존재하고 em은 존재하지 않으므로

tokens = [st, ##e, ##m] 으로 구성 됨