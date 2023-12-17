# 📃자연어 처리의 성능지표
### F1-Score
- F1-Score의 기초가 되는 Precision 과 Recall

**1. Recall(재현율)**
  - 실제 Positive 샘플 중 분류 모델이 Positive로 판정한 비율

  - 분류 모델이 실제 Positive 클래스를 얼마나 빠지지 않고 잘 잡아내는지를 나타냄
  - 
![Untitled](https://github.com/AYEOOON/AI-project/assets/101050134/e848f643-aa9b-4af8-81de-3de85c588bd8)

**2. Presision(정밀도)**
  - 분류 모델이 Positive로 판정한 것 중, 실제로 Positive인 샘플의 비율

  - Positive로 검출된 결과가 얼마나 정확한지를 보여줌

![Untitled (1)](https://github.com/AYEOOON/AI-project/assets/101050134/02fe6666-df12-4938-b9e9-b5177d229a76)

**3. F1-Score**
  - 분류 모델의 Precision과 Recall 성능을 동시에 고려하기 위해서 사용하는 지표

  - 예측 오류 개수만 관련되는 것이 아니라 발생한 오류의 종류도 관여

  - Precision과 Recall의 조화평균으로 정의

  - 0과 1사이의 값을 가지며 1에 가까울수록 분류 성능이 좋음을 나타냄

![Untitled (2)](https://github.com/AYEOOON/AI-project/assets/101050134/9837bb9d-2258-4dc4-ac17-625c081e0386)

### ROUGE(Recall-Orient-Understudy for Gisting Evaluation)
- 표면적 유사도 측정
  
- 모델이 생성한 요약본 혹은 번역본을 사람이 미리 만들어 놓은 참조본과 대조해 성능 점수를 계산
  
- ROUGE는 5개의 평가 지표가 있다.
    - ROUGE-N
    - ROUGE-L
    - ROUGE-W
    - ROUGE-S
    - ROUGE-SU
 
# 😂HuggingFace KoElectra로 NSMC 감성분석 Fine-tuning해보기
KoElectra-small을 이용해서 NSMC(Naver Sentiment Movie Corpus) 감성분석 모델을 학습해본다. 

학습은 Googel Colab(GPU)에서 Pytorch를 이용했다.

**1. 데이터셋 구조**

15만개의 train데이터와 5만개의 test데이터로 구성됐다. 

다만 일부 데이터가 NaN인 경우가 있으며, 중복된 데이터도 존재한다.

**2. 전처리**

HuggingFace에서 제공하는 AutoTokenizer를 이용했다.

KoElectra의 토크나이저를 쓰기 위해선 from_pretrained 함수의 인자로 ```monologg/koelectra-small-v2-discriminator``` 를 넣어주면 된다. 

**3. 모델**

Electra는 generator와 discriminator 2가지 모델을 학습하고 이중에서 discriminator를 사용한다. 

ElectraForSequenceClassification 에 ```“monologg/koelectra-small-v2-discriminator”``` 모델명을 줘서 모델을 가져온다.

**4. 학습결과**

4에폭을 학습했을 때, Train Set 정확도는 87.4%, Test Set 정확도도 86.87%를 달성했다.

1에폭 학습에 1시간 40분가량 소요됐다.

# 💸pytorch-2-0-bert-text-classification
2022년 12월 2일 PyTorch 2.0을 발표하였다. 

PyTorch 2.0은 더 나은 성능, 더 빠르고, 더 파이썬적이며, 이전처럼 역동적으로 유지하는 데 중점을 두었다. 

최신 PyTorch 2.0 기능을 사용하여 텍스트 분류를 위한 BERT 모델을 미세 조정하는 방법을 다룬다.

이 튜토리얼에서는 BANKING77 데이터세트에서 텍스트 분류 모델을 훈련하는 방법을 배운다.

**1. 환경 설정 및 PyTorch 2.0 설치**

첫 번째 단계는 변환기와 데이터 세트를 포함하여 PyTorch 2.0과 Hugging Face Libraries를 설치한다.

또한 PyTorch 2.0을 Trainer에 기본 통합하는 기능을 포함하는 기본 git 브랜치에서 최신 버전의 변환기를 설치한다. 


**2. 데이터 세트 로드 및 준비**

예제를 간단하게 유지하기 위해 BANKING77 데이터 세트에서 텍스트 분류 모델을 교육한다.

BANKING77 데이터 세트는 은행/금융 도메인에서 세분화된 의도(클래스) 세트를 제공한다. 


**3. Hugging Face Trainer를 사용하여 BERT 모델 미세 조정 및 평가 추론**

데이터 세트를 처리한 후 모델 학습을 시작할 수 있습니다. 우리는 bert-base-uncased 모델을 사용할 것이다.

첫 번째 단계는 Hugging Face Hub의 AutoModelForSequenceClassification 클래스를 사용하여 모델을 로드하는 것이다.

그러면 분류 헤드가 맨 위에 있는 사전 훈련된 BERT 가중치가 초기화한다.


**4. 실행 및 테스트 모델**

이 튜토리얼을 마무리하기 위해 몇 가지 예에 대해 추론을 실행하고 모델을 테스트한다. 

변환기 라이브러리의 파이프라인 방법을 사용하여 모델에 대한 추론을 실행한다.
