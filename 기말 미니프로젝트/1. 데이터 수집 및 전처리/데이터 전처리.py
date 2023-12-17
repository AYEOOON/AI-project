from konlpy.tag import Okt  # 한국어 텍스트 처리를 위해 Konlpy 라이브러리의 Okt 클래스를 가져옵니다.
import pandas as pd  # 데이터 조작을 위해 pandas를 가져옵니다.
import re  # 정규 표현식을 위해 re를 가져옵니다.
import pickle  # 직렬화를 위해 pickle을 가져옵니다.

# 한국어 텍스트 처리를 위해 Okt를 초기화합니다.
okt = Okt()

# 노래 데이터가 있는 피클 파일에서 DataFrame을 로드합니다. ('lyric', 'title', 'artist' 등의 열을 포함한다고 가정합니다.)
melon = pd.read_pickle('/songlist(1000).pkl')

# 텍스트 데이터를 정리하기 위한 전처리 함수입니다.
def preprocessing(x):
    x = x.lower()  # 텍스트를 소문자로 변환합니다.
    x = x.replace('\n', ' ')  # 줄 바꿈 문자를 공백으로 대체합니다.
    return x

# DataFrame의 'lyric' 열에 전처리 함수를 적용합니다.
melon.lyric = melon.lyric.apply(lambda x: preprocessing(x))

# Okt의 명사 추출기를 사용하여 가사를 단어로 토큰화합니다.
melon['word'] = melon['lyric'].apply(okt.nouns)

# 불필요한 열을 DataFrame에서 삭제합니다.
melon.drop(['pages', 'title', 'artist'], axis=1, inplace=True)

# 각 행에 대한 고유한 단어를 추출하고 처음 5개의 고유한 단어를 유지합니다.
unique_words_list = []
for w in melon['word']:
    unique_words = list(set(word for word in w if len(word) > 1))  # 1자 이상인 고유한 단어를 필터링합니다.
    unique_words_list.append(unique_words[:5])  # 처음 5개의 고유한 단어를 유지합니다.

# 추출된 고유한 단어로 새로운 'filtered_word' 열을 만듭니다.
melon['filtered_word'] = unique_words_list

# 'filtered_word' 열을 형식화하여 대괄호를 제거하고 문자열로 변환합니다.
melon['filtered_word'] = melon['filtered_word'].apply(lambda x: str(x).replace('[', '').replace(']', ''))
melon.drop(['word'], axis=1, inplace=True)

# 'lyric'과 'word' 열을 포함하는 DataFrame을 생성합니다.
data = {'lyric': melon['lyric'], 'word': melon['filtered_word']}
df = pd.DataFrame(data)

# 가사와 필터된 단어를 텍스트 파일로 저장하는 함수입니다.
def save_lyrics_to_text_file(df, filename, num_lyrics=200):
    with open(filename, 'w', encoding='utf-8') as file:
        for idx in range(num_lyrics):
            # 각 행에서 가사와 필터된 단어를 추출하여 텍스트 파일에 작성합니다.
            lyric = df.loc[idx % len(df)]['lyric']
            filter_word = df.loc[idx % len(df)]['word']
            line = f"{lyric} | {filter_word}\n"  # 가사와 필터된 단어를 '|'로 구분하여 한 줄에 저장합니다.
            file.write(line)

# 저장할 텍스트 파일의 파일명입니다.
file_name = 'hiphop.txt'

# save_lyrics_to_text_file 함수를 호출하여 데이터를 텍스트 파일로 저장합니다.
save_lyrics_to_text_file(df, file_name, num_lyrics=200)
