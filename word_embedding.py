# %%
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
from tqdm import tqdm
# %%
data_top1_df = pd.read_csv('./data/mutandard_top1.csv', encoding='cp949', index_col=0)
data_top2_df = pd.read_csv('./data/mutandard_top2.csv', encoding='cp949', index_col=0)

def preprocessing(data):
    data['height'] = [int(height.strip().split('c')[0]) for height in data['height']]
    data['weight'] = [int(weight.strip().split('k')[0]) for weight in data['weight']]
    
    return data

data_top1_df = preprocessing(data_top1_df)
data_top2_df = preprocessing(data_top2_df)
# %%
