# %%
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models import Word2Vec
from gensim.models import FastText
from konlpy.tag import Okt
from tqdm import tqdm
# %%
data_top1_df = pd.read_csv('./data/mutandard_top1.csv', encoding='cp949', index_col=0)
data_top2_df = pd.read_csv('./data/mutandard_top2.csv', encoding='cp949', index_col=0)
# data_top3_df = pd.read_csv('./data/mutandard_top3.csv', encoding='cp949', index_col=0)

def preprocessing(data):
    data['height'] = [int(height.strip().split('c')[0]) for height in data['height']]
    data['weight'] = [int(weight.strip().split('k')[0]) for weight in data['weight']]
    
    return data

data_top1_df = preprocessing(data_top1_df)
data_top2_df = preprocessing(data_top2_df)
# data_top3_df = preprocessing(data_top3_df)

# %%
## 결측값 존재하는 행 제거
data_top1_df = data_top1_df.dropna(how = 'any') # Null 값이 존재하는 행 제거
## 정규 표현식을 통한 한글 외 문자 제거
data_top1_df['content'] = data_top1_df['content'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
## 불용어 정의
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

##  형태소 분석기 OKT를 사용한 토큰화 작업 (다소 시간 소요)
okt = Okt()

tokenized_data = []
for sentence in tqdm(data_top1_df['content']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    tokenized_data.append(stopwords_removed_sentence)

# %%
## 리뷰 길이 분포 확인
print('리뷰의 최대 길이 :',max(len(review) for review in tokenized_data))
print('리뷰의 평균 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))

plt.hist([len(review) for review in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

# %%
## Word2Vec으로 토큰화 된 네이버 영화 리뷰 데이터 학습
model = Word2Vec(sentences = tokenized_data,
                 vector_size = 128, # 임베딩 차원
                 window = 5, # 앞뒤로 몇개 확인
                 min_count = 3, # 단어의 최소 노출수
                 workers = 4,
                 sg = 0)

## 완성된 임베딩 매트릭스의 크기 확인
print(model.wv.vectors.shape) # 총 703개의 단어가 존재하며 각 단어는 128차원으로 구성
print(model.wv.most_similar("사이즈")) # '사이즈'와 유사한 단어들

# %%
## FastText
model_fasttext = FastText(sentences = tokenized_data,
                 vector_size = 128,
                 window = 5,
                 min_count = 3,
                 workers = 4,
                 sg = 0)

## 완성된 임베딩 매트릭스의 크기 확인
print(model_fasttext.wv.vectors.shape) # 총 703개의 단어가 존재하며 각 단어는 128차원으로 구성
print(model.wv.most_similar("사이즈")) # '사이즈'와 유사한 단어들

# %%
