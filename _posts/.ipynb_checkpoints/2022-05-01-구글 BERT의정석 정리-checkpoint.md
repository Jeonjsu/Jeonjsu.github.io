---
layout: post
title: 구글 BERT의 정석 정리 6장
categories: [NLP]
tags: [NLP]
description: 텍스트 요약/ 추출
---
- 텍스트 요약 / BERT 파인튜닝
- BERT를 활용한 추출/생성 요약
- ROUGE 평가지표

### 6.1 텍스트 요약 / 파인튜닝

텍스트 요약은 아래 2개의 TASK로 구분됨

- 추출요약

전체 문단의 중요 문장만을 추출

- 생성요약

전체 문단중 의역하여 새로운 문장을 생성

### 6.2 BERT 활용한 추출/생성 요약

6.2.1 추출요약 (BERTSUM)

BERT 모델에 사용하기 위한 INPUT 데이터 형태 이며 문장별 시작부분에 CLS를 추가

입력 : [CLS] Paris is a beatiful city [SEP] [CLS] I love Paris [SEP]

토큰 : E_cls + E_paris ...                                             + E_sep

세그 : E_A + E_A +                               +E_B +         + E_B

위치 : E_0 + E_1 +                                                    + E_10

위 3가지 종류의 벡터값들이 트랜스포머 인코더를 통과하여 

출력 : R_[CLS] , R_Paris , ...   R_[SEP] 값을 출력하고

전체문장이 담겨있는 R_[CLS] 벡터를 이진분류 함. 이 벡터값들이 아래의 input값으로 활용

분류기에 트랜스포머 / LSTM 을 활용 가능함

트랜스포머

- L개의 트랜스포머 인코더의 마지막 출력값의 hidden layer h^L 을 이진분류기에 통과

LSTM

- LSTM 을 통과시켜 각 문장별 hidden layer를 얻고, 각각을 이진분류기에 통과

6.2.2 생성요약 (BERTSUMABS , bert for abstactive summarization)

- 트랜스 포머의 인코더 디코더 구조 사용
- 인코더 - BERTSUM , 디코더 - 무작위로 초기화됨
    - 인코더 , 디코더의 optimizer를 구분하여 사용
    - 인코더는 lr 작게, 디코더는 lr 크게

참고) [https://wikidocs.net/31379](https://wikidocs.net/31379)

### 6.3 ROUGE 평가지표

텍스트 요약 태스트 평가 지표 ROUGE(Recall-Oriented Understudy for Gisting Evaluation)

- ROUGE-N
- ROUGE-L

6.3.1 ROUGE-N 이해

후보요약 (예측한 요약) 과 참조요약 (실제요약) 간의 n-gram 재현율

재현율  = 겹치는 n-gram 수 / 참조요약의 n-gram 수

후보요약(예측) - Machine learning is seen as a subset of artificial intelligence.

참조요약(실제) - Machine learning is a subset of artificial intelligence.    

Rouge 2를 계산하면 

후보요약 : (machine , learning ) ( learning is )  ... ( artificial intelligence) 

참조요약 : (machine , learning ) (learning is ) ... (artificial intelligence) 

6/7 이 계산됨

6.3.2 ROUGE-L 이해

가장 긴 공통시퀀스 LCS ( longest common subsequence )를 사용함

$R_{lcs} = \frac{LCS(후보,참조)}{참조요약의 전체 단어 수}$ 

$P_{lcs} = \frac{LCS(후보, 참조)}{후보요약의 전체 단어 수}$

$F_{lcs} = \frac{(1+b^2)RP}{R+b^2P}$

참고 ) [https://huffon.github.io/2019/12/07/rouge/](https://huffon.github.io/2019/12/07/rouge/)