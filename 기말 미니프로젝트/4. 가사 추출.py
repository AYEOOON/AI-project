from transformers import GPT2LMHeadModel, GPT2Tokenizer, PreTrainedTokenizerFast

def generate_lyrics_from_keywords(model_path, keyword_list, max_length_per_keyword=30, max_total_length=500, max_lines=20):
    # GPT-2 모델과 토크나이저를 불러오는 함수 정의

    # PreTrainedTokenizerFast를 사용하여 KoGPT-2 모델의 토크나이저를 불러옵니다.
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                      bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                      pad_token='<pad>', mask_token='<mask>')
    # GPT-2LMHeadModel을 사용하여 KoGPT-2 모델을 불러옵니다.
    model = GPT2LMHeadModel.from_pretrained(model_path)

    generated_lyrics = []  # 생성된 가사를 저장할 리스트
    total_length = 0  # 생성된 가사의 총 길이
    total_lines = 0  # 생성된 가사의 총 줄 수

    for keyword in keyword_list:
        # 키워드 리스트의 각 키워드를 활용하여 가사를 생성합니다.
        input_text = keyword
        input_ids = tokenizer.encode(input_text, return_tensors='pt')

        # GPT-2 모델을 사용하여 가사를 생성합니다.
        generated = model.generate(input_ids, max_length=max_length_per_keyword, num_return_sequences=1,
                                   pad_token_id=tokenizer.eos_token_id)

        # 생성된 가사를 토크나이저를 사용하여 디코딩하고 리스트에 추가합니다.
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        words = generated_text.split()  # 공백을 기준으로 단어를 분리합니다.

        # 각 키워드마다 15단어마다 줄 바꿈하여 가사를 생성합니다.
        new_lyrics = ''
        for word in words:
            if len(new_lyrics.split()) >= 10:  # 15단어 이상이 되면 줄 바꿈합니다.
                generated_lyrics.append(new_lyrics)
                new_lyrics = ''
                total_lines += 1
                if total_lines >= max_lines:  # 설정한 최대 줄 수를 넘으면 가사 생성을 종료합니다.
                    return '\n'.join(generated_lyrics)
            new_lyrics += word + ' '

        if new_lyrics:  # 남은 내용이 있다면 마지막 줄에 추가합니다.
            generated_lyrics.append(new_lyrics)
            total_lines += 1
            if total_lines >= max_lines:  # 설정한 최대 줄 수를 넘으면 가사 생성을 종료합니다.
                return '\n'.join(generated_lyrics)

    # 생성된 가사를 합쳐서 하나의 새로운 가사로 반환합니다. (줄 바꿈 포함)
    combined_lyrics = '\n'.join(generated_lyrics)
    return combined_lyrics

# 모델 저장 경로와 키워드 리스트 지정
model_save_path = 'lyrics_model'
keyword_list = input('키워드를 적어주세요(5단어 이하): ')

# 각 키워드별 가사 생성 및 합치기 (줄 바꿈 포함)
result_lyrics = generate_lyrics_from_keywords(model_save_path, keyword_list, max_lines=20)

# 생성된 가사 출력
print("입력 키워드:", keyword_list)
print("생성된 가사:")
print(result_lyrics)
