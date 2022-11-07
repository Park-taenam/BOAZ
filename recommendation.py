'''
Content : Recommendation System
Author : Taenam
'''
# %% Import
from nltk.corpus import stopwords
import datetime
from tqdm import tqdm_notebook, tqdm   # for문 진행상황 눈으로 확인 (loading bar)
from PIL import Image
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import word_tokenize
import nltk
from konlpy.tag import *   # 모든 형태소분석기 import 하기
import seaborn as sns
import pandas as pd
import numpy as np
import os
import sys
import gc
import re
import io
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

# %% Road and preprocessing data
# data = pd.read_pickle('data.pkl')
data_top1 = pd.read_csv('./data/mutandard_top1.csv',
                        encoding='cp949', index_col=0)

# 데이터 전처리 함수


def preprocessing(data):
    # height, weight 숫자로 변경
    data['height'] = [int(height.strip().split('c')[0])
                      for height in data['height']]
    data['weight'] = [int(weight.strip().split('k')[0])
                      for weight in data['weight']]

    # 사이즈 평가
    data["size_eval"] = data["size_eval"].replace('보통이에요', '0', regex=True)
    data["size_eval"] = data["size_eval"].replace('커요', '1', regex=True)
    data["size_eval"] = data["size_eval"].replace('작아요', '-1', regex=True)
    data["size_eval"] = pd.to_numeric(data["size_eval"])

    return data


data_top1 = preprocessing(data_top1)

# %% KNN + KNeighborsClassifier
## KNN Classifier
X = data_top1.loc[:, ['height', 'weight']].values
y = data_top1.loc[:, 'size'].values

scaler_x = StandardScaler()
scaler_x.fit(X)
X_scaled = scaler_x.transform(X)

# plt.scatter(pd.DataFrame(X)[0], pd.DataFrame(X)[1])
# plt.show()
# plt.scatter(pd.DataFrame(X_scaled)[0], pd.DataFrame(X_scaled)[1])
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)  # 하이퍼파라미터 조정 필요
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %%
## For each user
pivot = data_top1.pivot_table("size_eval", index="user", columns="size")  # 점수 부여 필요

# 점수 부여하는 함수 구현 필요 **********************************************












## With categorization

# %%
