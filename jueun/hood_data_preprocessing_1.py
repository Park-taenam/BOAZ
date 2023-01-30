# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# �ð�ȭ ����
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
from PIL import Image
from matplotlib import rc

# �׷������� �ѱ� ��Ʈ ������ ������ ���� ��ó(���� �۲� ����)
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

from konlpy.tag import Kkma #���¼Һм�
from konlpy.tag import Okt #���¼Һм�
import soynlp; from soynlp.normalizer import * #����ȭ
# from pykospacing import Spacing------------------>�����!
# spacing = Spacing()------------------------------>�����!

from konlpy.tag import Komoran, Hannanum, Kkma, Okt
komoran = Komoran(); hannanum = Hannanum(); kkma = Kkma(); okt = Okt();

# %%
def hood_crawlingdataconcat():
    df = pd.DataFrame()
    for i in range(1,21,1):
        try:
            df_before = pd.read_csv('data/hood_'+str(i)+'page.csv') #ũ�Ѹ����� �ҷ�����
            print('df',str(i),': ',df_before.shape,end = ',')
        except:
            df_before = pd.DataFrame() #����ũ�Ѹ����� ���������� �� df
            print('df',str(i),': ',df_before.shape)
            
        df = pd.concat([df,df_before]) #��ġ��
            

            
    print('�����ģdf : ', df.shape)
    df.drop_duplicates(subset = None,keep = 'first', inplace = True,ignore_index = True)
    print('�ߺ�����df : ', df.shape)#��ü���� ���� �ߺ�����
    #����,Ű,������ ��� null�ƴϰ� ��ġ���� �ϳ��� null�ƴ� �� ����
    df = df[~df['gender'].isnull()&~df['height'].isnull()&~df['weight'].isnull()&
                                      (~df['����'].isnull()|
                                       ~df['����ʺ�'].isnull()|~df['�����ܸ�'].isnull()|
                                       ~df['�Ҹű���'].isnull())]
    print('������� data:{}'.format(df.shape))
    print('-'*10)
    return df


def crawlingdataprocessing(df):
    #kg,cm���� �� ��ġŸ������ ����
    df['gender']=df['gender'].apply(lambda x:x.replace("����","1"))  
    df['gender']=df['gender'].apply(lambda x:x.replace("����","0"))  
    df["height"]=df["height"].replace('cm','',regex = True)
    df["weight"]=df["weight"].replace('kg','',regex = True)
    df= df.astype({'gender':'int','height':'float','weight':'float'})
    print('df��ó���Ϸ�')
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
