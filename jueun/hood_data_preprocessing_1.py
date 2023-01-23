# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 시각화 관련
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
from PIL import Image
from matplotlib import rc

# 그래프에서 한글 폰트 깨지는 문제에 대한 대처(전역 글꼴 설정)
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings(action='ignore') 

import matplotlib
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

#nlp
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
from selenium import webdriver
from tqdm.notebook import tqdm
import time, urllib
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
keys = Keys()
import re, os, sys, json

from hanspell import spell_checker

from konlpy.tag import Kkma #형태소분석
from konlpy.tag import Okt #형태소분석
import soynlp; from soynlp.normalizer import * #정규화
# from pykospacing import Spacing------------------>써야함!
# spacing = Spacing()------------------------------>써야함!

from konlpy.tag import Komoran, Hannanum, Kkma, Okt
komoran = Komoran(); hannanum = Hannanum(); kkma = Kkma(); okt = Okt();

# %%
def hood_crawlingdataconcat():
    df = pd.DataFrame()
    for i in range(1,21,1):
        try:
            df_before = pd.read_csv('data/hood_'+str(i)+'page.csv') #크롤링파일 불러오기
            print('df',str(i),': ',df_before.shape,end = ',')
        except:
            df_before = pd.DataFrame() #아직크롤링되지 않은파일은 빈 df
            print('df',str(i),': ',df_before.shape)
            
        df = pd.concat([df,df_before]) #합치기
            

            
    print('모두합친df : ', df.shape)
    df.drop_duplicates(subset = None,keep = 'first', inplace = True,ignore_index = True)
    print('중복제거df : ', df.shape)#전체열이 같은 중복제거
    #성별,키,몸무게 모두 null아니고 수치종류 하나라도 null아닌 것 필터
    df = df[~df['gender'].isnull()&~df['height'].isnull()&~df['weight'].isnull()&
                                      (~df['총장'].isnull()|
                                       ~df['어깨너비'].isnull()|~df['가슴단면'].isnull()|
                                       ~df['소매길이'].isnull())]
    print('최종사용 data:{}'.format(df.shape))
    print('-'*10)
    return df


def crawlingdataprocessing(df):
    #kg,cm제거 및 수치타입으로 변경
    df['gender']=df['gender'].apply(lambda x:x.replace("남성","1"))  
    df['gender']=df['gender'].apply(lambda x:x.replace("여성","0"))  
    df["height"]=df["height"].replace('cm','',regex = True)
    df["weight"]=df["weight"].replace('kg','',regex = True)
    df= df.astype({'gender':'int','height':'float','weight':'float'})
    print('df전처리완료')
    print('-'*10)
    return df
# %%
df = hood_crawlingdataconcat()
df = crawlingdataprocessing(df)
df.to_pickle('data/crawlingdata_preprocess_done.pkl')#,encoding="UTF-8", index=False)
# %%
# df = pd.read_pickle('data/crawlingdata_preprocess_done.pkl')
# df.head()
# %%
