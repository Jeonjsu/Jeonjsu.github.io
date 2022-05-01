---
layout: post
title: 구글 BERT의 정석 정리 5장
categories: [NLP]
tags: [NLP]
description: 파생모델 2 ; 지식증류
---
## BERT 파생모델 2 : 지식 증류 기반

사전학습 된 BERT를 사용하는 데에는 리소스적 제한이 큼. 이를 해결하기 위해 대형 BERT에서 소형 BERT로 지식 증류를 사용. 

- 지식 증류란 무엇인가?
- DistilBERT 정리
- TinyBERT 정리
- BERT에서 신경망으로 지식 전달

### 5.1 지식증류란?

대형모델의 동작을 재현하기 위해 소형모델을 학습시키는 모델압축기술이며 teacher-student learning이라고도 함.

일반적으로 신경망을 통과 후 소프트맥스 함수를 통해 target값의 확률 분포를 얻음.

예를들어 I completed writing my [] . 이란 문장에서 마지막 단어의 예측 분포가 

homework  0.6

book  0.15

cake  0.05

assignment  0.15

car 0.05

이었다고 하면 정답값인 homework 외에 book과 assignment 가 0.15로 상대적으로 cake와 car에 비해 확률값이 높게 나온다. 이것을 dark knowledge라 함. 

그러나 만약 homework가 0.999 라고 하면, 위와같은 dark knowledge를 알 수 없게됨. 그래서 소프트템퍼러쳐를 사용하여 확률분포 평활화를 함

`소프트맥스 템퍼러쳐 : 𝑞𝑖 =exp(𝑧𝑖/𝑇)/ ∑𝑗exp(𝑧𝑗/𝑇)`

5.1.1 학생 네트워크 학습

![aa.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ec140089-cd35-4a9f-b47f-2809f370b297/aa.png)

Loss = a*Distillation Loss + b*Student Loss

Distillation Loss : Teacher 모델에서 예측값과, 레이어를 더 단순화한 Student 모델에서 예측값의 차이

Student Loss : Student 모델에서 Softmax 예측값 ( 0.6 , 0.1 , ...) 과 hard target ( 1, 0 , ... 0 ) 의 차이

위 Loss 를 최소화 하도록 학습

### 5.2 DistillBERT

기본 BERT에 비해 60% 빠르며 40% 작다고 알려짐

5.2.1 교사-학생 아키텍쳐

기본 BERT는 masking된 부분의 단어 확률분포를 알 수 있음. 이것을 학생 BERT에 전달해야함

- 교사 BERT는 1억 1000만개의 parameter 존재. 학생 BERT는 6,600만개의 parameter 존재
- 은닉 상태의 차원을 줄이는 것 보다 레이어의 수를 줄이는 것이 더 유효함

5.2.2 학생 BERT 학습

RoBERTa의 학습방법 몇가지를 차용함

- 동적 masking

기본 bert에서 masking된 부분에서 단어들의 확률분포가 나옴. 

이것을 학생 bert에서의 예측값과 차이를 통해 distillation loss를 계산.

그 후 학생 bert에서 hard target과 soft target 차이를 통해 student loss 계산.

마지막으로 교사 bert와 학생 bert 출력값 간의 코사인 임베딩 loss를 계산함.

학생 bert는 교사 bert의 약 97% 성능을 제공하고, 60% 더 빠름

### 5.3 TinyBERT

- 입력(임베딩), 인코더(트래스포머), 출력 세가지 레이어에서 지식 전달.
- data augmentation을 통해 특정 task에 특화된 학생 모델 생성

5.3.1트랜스포머 layer 

- 어텐션 기반
    - 교사-학생 어텐션 행렬사이의 MSE를 최소화
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e58f180f-8282-4bd9-bfec-9a2ea42ff18d/Untitled.png)
    
- hidden layer 기반
    - 교사-학생 사이의 hidden layer 내의 PARAMETER 들의 MSE 최소화
    - 일반적으로 학생 hidden layer의 차원이 낮으므로, w_h를 곱해 차원을 맞춤
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/826126b9-1116-4df7-93c6-eaa0bfebc02e/Untitled.png)
    

5.3.2 임베딩 layer

- 트랜스포머의 hidden layer와 유사함
- 차원을 맞춰줌

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/114f0996-9c4c-4480-a9b5-73dd648faa05/Untitled.png)

5.3.3 예측 layer

- 증류 손실 사용

5.3.4 TinyBERT 학습

2단계 학습을 진행

- 일반 증류
- task 특화 증류

일반증류?

5.3.1~5.3.3 에 설명된 방식대로, BERT에서 tinyBERT를 생성

task 특화 증류?

- Bert-base를 특정 task에 맞게 fine-tuning함.
- fine-tuning된 BERT를 tinyBERT로 지식 증류

fine-tuning 된 것을 증류할 때, data augmentation을 함.

data-agumentation?

만약 Paris is a beautiful city 라는 문장이 있다고 하면

위 문장을 토큰화 하면

X = [Paris, is, a, beautiful, city] 가 되고

위 단어 list의 모든 단계에서 아래와 같은 task가 수행됨

i = 0 에서 단일단어이면 [MASK] 로 토큰을 교체

X_masked = [ [MASK] , is, a, beautiful, city]

이때 MASK에 올 단어를 BERT-base가 확률분포로 예측하는데 K=3으로 설정하여 3개의 단어를 출력했다고 하면

candidate = [Paris , it , that] 임.

이때 균등분포 p~U(0,1) 로 p를 샘플링하여 임계값인 0.4보다 낮게 나오면 candidate의 단어로 교체함

X_masked = [it, is, a, beautiful, city] 

이 X_masked를 data_aug 리스트에 추가하고 위 단계를 N번 반복해 더 많은 데이터를 얻고, 증식된 데이터로 fine-tuning함.

이렇게 생성된 TinyBERT는 BERT-base에 비해 효율면에서 96% 좋고, 7.5배 작으며, 9.4배 빠름

### 5.4 BERT에서 신경망으로 지식 전달