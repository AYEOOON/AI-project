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

# 35개의 데이터세트에서 사용가능한 라벨목록 중 마지막 단어를 출력하는 코드
waveform_last, *_ = train_set[-1]
ipd.Audio(waveform_last.numpy(), rate=sample_rate)  # 오디오 파일을 출력하기 위한 Ipython 내장라이브러리


# 3. 데이터 형식 지정
# 파형의 경우 분류 능력을 너무 많이 잃지 않으면서 더 빠른 처리를 위해 오디오를 다운샘플링합니다. 
# 이 튜토리얼의 경우 오디오에 단일 채널을 사용하무로 여기서는 필요하지 않습니다. 
new_sample_rate = 8000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)

ipd.Audio(transformed.numpy(), rate=new_sample_rate)

# 레이블 목록의 인덱스를 사용하여 각 단어를 인코딩합니다. 
def label_to_index(word):
    # 35개의 오디오 라벨에서 입력된 단어의 인덱스를 반환
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # 오디오 라벨에서 입력된 인덱스에 해당하는 단어를 반환
    # label_to_index함수의 반대되는 과정
    return labels[index]


word_start = "yes"
index = label_to_index(word_start)
word_recovered = index_to_label(index)

print(word_start, "-->", index, "-->", word_recovered)


# tensor padding하는 함수
def pad_sequence(batch):
    # 0으로 채워 일괄 처리의 모든 텐서를 동일한 길이로 만듦.
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.) # 원하는대로 padding을 줄 수 있는 함수
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # 데이터 튜플의 형식:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # 각각의 리스트 안에 넣고, 라벨을 인덱스로 인코딩함
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


batch_size = 256

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

train_loader = torch.utils.data.DataLoader(  # DataLoader: 데이터셋을 읽어와서 배치 단위로 데이터를 불러옴. 이를 통해 모델 학습을 더 효율적으로 진행
    train_set,
    batch_size=batch_size,   # batch_size: DataLoader가 반환할 배치(batch) 크기입니다. 기본값은 1
    shuffle=True,  # shuffle: 데이터셋을 무작위로 섞을지 여부를 결정하는 파라미터, True로 설정하면 매 에포크마다 데이터셋이 섞임
    collate_fn=collate_fn, # collate_fn: 배치(batch) 단위로 데이터를 처리하는 함수
    num_workers=num_workers, # num_workers: 데이터를 불러올 때 사용할 프로세스(worker) 수. 기본값은 0
    pin_memory=pin_memory, # pin_memory: True로 설정하면, 반환된 배치 데이터는 CUDA 호환 GPU 메모리에 고정됩니다. 기본값은 False
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)


# 4. 네트워크 정의
# 



# 5. 네트워크 훈련 및 테스트
# 훈련 데이터를 모델에 공급하고, 역방향 전달 및 최적화 단계를 수행하는 훈련함수
def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())

# 2개의 에포크를 테스트 하는 코드
log_interval = 20
n_epoch = 2

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []

# The transform needs to live on the same device as the model and the data.
transform = transform.to(device)
with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        test(model, epoch)
        scheduler.step()

# Let's plot the training loss versus the number of iteration.
# plt.plot(losses);
# plt.title("training loss");

# 네트워크는 2개의 에포크 후에는 테스트 세트에서 65% 이상 정확해야하며, 21개의 에포크 후에는 85% 이상 정확해야합니다. 

# 나의 목소리 녹음 후 돌려본 뒤 인식 결과
def record(seconds=1):

    from google.colab import output as colab_output
    from base64 import b64decode
    from io import BytesIO
    from pydub import AudioSegment

    RECORD = (
        b"const sleep  = time => new Promise(resolve => setTimeout(resolve, time))\n"
        b"const b2text = blob => new Promise(resolve => {\n"
        b"  const reader = new FileReader()\n"
        b"  reader.onloadend = e => resolve(e.srcElement.result)\n"
        b"  reader.readAsDataURL(blob)\n"
        b"})\n"
        b"var record = time => new Promise(async resolve => {\n"
        b"  stream = await navigator.mediaDevices.getUserMedia({ audio: true })\n"
        b"  recorder = new MediaRecorder(stream)\n"
        b"  chunks = []\n"
        b"  recorder.ondataavailable = e => chunks.push(e.data)\n"
        b"  recorder.start()\n"
        b"  await sleep(time)\n"
        b"  recorder.onstop = async ()=>{\n"
        b"    blob = new Blob(chunks)\n"
        b"    text = await b2text(blob)\n"
        b"    resolve(text)\n"
        b"  }\n"
        b"  recorder.stop()\n"
        b"})"
    )
    RECORD = RECORD.decode("ascii")

    print(f"Recording started for {seconds} seconds.")
    display(ipd.Javascript(RECORD))
    s = colab_output.eval_js("record(%d)" % (seconds * 1000))
    print("Recording ended.")
    b = b64decode(s.split(",")[1])

    fileformat = "wav"
    filename = f"_audio.{fileformat}"
    AudioSegment.from_file(BytesIO(b)).export(filename, format=fileformat)
    return torchaudio.load(filename)


# Detect whether notebook runs in google colab
if "google.colab" in sys.modules:
    waveform, sample_rate = record()
    print(f"Predicted: {predict(waveform)}.")
    ipd.Audio(waveform.numpy(), rate=sample_rate)


# 결과1
# left라고 했을 때 six가 나옴
Recording started for 1 seconds.
Recording ended.
Predicted: six.

# 결과2
# six라고 하면 yes가 나옴
Recording started for 1 seconds.
Recording ended.
Predicted: yes.
