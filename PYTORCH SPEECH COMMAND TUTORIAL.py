# 음성인식과정
# 사람의 목소리를 특정할 수 있는 주파수 대역을 뽑음
# => 음성으로 판단되는 변화가 존재할 때 이를 음성인식에 사용
# => 추출된 음성데이터에서 특징추출
# => 녹음된 음성을 초당 50회 정도로 쪼개어 음성이 변화하는 특징을 수치화하여 분석하기 쉽게 전처리
# => 이런 각각의 데이터간의 변화, 즉 "특징벡터"를 구함
# => 쪼개진 데이터가 어떤 음소에 매칭되는지를 만들어 내는 과정인 음향 모델링을 진행
# => 딥러닝에 필요한 학습데이터를 기반으로 앞서 만들어진 특징 벡터와의 비교를 통해 음성 데이터의 변화가 개별 음소에 매칭될 확률을 학습


# 1. 라이브러리 설치
# 먼저 웹사이트에 지침에 따라 설치할 수 있는 torchaudio와 같은 일반적인 토치 패키지를 가져옵니다. 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt
import IPython.display as ipd

from tqdm import tqdm


# 2. 데이터세트 가져오기
# torchaudio를 사용하여 데이터세트를 다운로드하고 표현
# 35개 명령의 데이터 세트인 SpeechCommands를 사용
from torchaudio.datasets import SPEECHCOMMANDS
import os


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):  # 파일을 읽는 함수
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


# SPEECHCOMMANDS 데이터 세트의 데이터 포인트는 파형(오디오 신호), 샘플 속도, 발화(레이블), 화자의 ID, 발화 수로 구성된 튜플입니다.

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.plot(waveform.t().numpy());

# 음성파일을 torchaudio로 읽으면 음성 데이터와 일정한 시간간격으로 음압을 측정하는 주파수인 sampling rate를 반환
# 튜토리얼에 있는 음성데이터의 결과를 보면[1,16000]으로 1은 채널의 개수(녹음한 마이크의 개수)를 의미하고, 16000는 데이터의 길이를 의미
# 가로축은 시간, 세로축은 음압을 나타낸다. 음압이 큰 부분에 어떤 목소리, 큰 소리가 있음을 추측할 수 있다. 

# 데이터세트에서 사용가능한 라벨 목록을 찾아보는 코드
labels = sorted(list(set(datapoint[2] for datapoint in train_set)))

