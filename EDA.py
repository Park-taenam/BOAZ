# %%
import numpy as np
import pandas as pd
import mlxtend

import warnings
warnings.filterwarnings('ignore')
# %%
data_top1 = pd.read_csv('./data/mutandard_top1.csv', encoding='cp949', index_col=0)
data_top2 = pd.read_csv('./data/mutandard_top2.csv', encoding='cp949', index_col=0)
data_top3 = pd.read_csv('./data/mutandard_top3.csv', encoding='cp949', index_col=0)
# %%
print("키 종류 수 : ", data_top1['height'].nunique())
print("몸무게 종류 수 : ", data_top1['weight'].nunique())
print("상품 종류 수 : ", data_top1['item'].nunique())
print("사이즈 종류 수 : ", data_top1['size'].nunique())
print("평점 종류 수 : ", data_top1['star'].nunique())
print("리뷰 종류 수 : ", data_top1['content'].nunique())
# %%
def preprocessing(data):
    data['height'] = [int(height.strip().split('c')[0]) for height in data['height']]
    data['weight'] = [int(weight.strip().split('k')[0]) for weight in data['weight']]
    
    return data

data_top1 = preprocessing(data_top1)
data_top2 = preprocessing(data_top2)
data_top3 = preprocessing(data_top3)

# create pickle
data = pd.concat([data_top1, data_top2, data_top3], axis=0)
data.to_pickle('data.pkl')

data_pkl = pd.read_pickle('data.pkl')
data_pkl
# %%
