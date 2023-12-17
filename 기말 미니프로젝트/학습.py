# 필요한 패키지 설치
!pip install fastai==2.2.5
!pip install transformers

# 필요한 라이브러리 및 모듈 임포트
import torch
import transformers
from transformers import AutoModelWithLMHead, PreTrainedTokenizerFast
from fastai.text.all import *
import fastai
from nltk.tokenize import word_tokenize
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader

# 가사 데이터셋을 이용한 KoGPT2 모델 학습

# Dataset 클래스를 상속하여 가사 데이터셋을 생성하는 클래스
class LyricsDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=100):
        self.lyrics = []  # 가사 저장 리스트
        self.keywords = []  # 키워드 저장 리스트
        self.tokenizer = tokenizer  # 토크나이저
        self.max_length = max_length  # 최대 길이 설정

        # 가사 데이터셋 파일 읽기
        with open(data_path, 'r', encoding='utf-8') as file:
            data = file.readlines()

        # 데이터 전처리: 가사와 키워드 분리하여 리스트에 저장
        for line in data:
            line = line.strip().split('|')
            self.lyrics.append(line[0])  # 가사
            self.keywords.append(line[1])  # 키워드

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, idx):
        return self.keywords[idx], self.lyrics[idx]

# 가사 데이터셋 파일 경로
lyrics_file_path = '/content/lyrics_with_filters(3).txt'

# 모델 및 토크나이저 불러오기
model_name_or_path = 'skt/kogpt2-base-v2'
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

# 데이터셋 생성
lyrics_dataset = LyricsDataset(lyrics_file_path, tokenizer)
data_loader = DataLoader(lyrics_dataset, batch_size=4, shuffle=True)

# 모델 학습
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU 사용 여부 확인
model.to(device)  # 모델을 GPU로 이동
model.train()  # 학습 모드로 설정
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 옵티마이저 설정

num_epochs = 5  # 에폭 수
for epoch in range(num_epochs):
    total_loss = 0
    for keywords, lyrics in data_loader:
        # 토큰화 및 최대 길이 설정
        inputs = tokenizer(lyrics, return_tensors='pt', padding=True, truncation=True, max_length=64)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss  # 손실 계산
        loss.backward()  # 역전파
        optimizer.step()  # 옵티마이저 업데이트
        optimizer.zero_grad()  # 그래디언트 초기화
        total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_loss:.4f}')

# 학습된 모델 저장
model.save_pretrained('lyrics_model')
tokenizer.save_pretrained('lyrics_model')
