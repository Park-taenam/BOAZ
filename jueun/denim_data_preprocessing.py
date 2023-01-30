
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
def denim_crawlingdataconcat():
    df = pd.DataFrame()
    for i in range(1,21,1):
        try:
            df_before = pd.read_csv('data/denim_page'+str(i)+'.csv') #ũ�Ѹ����� �ҷ�����
            print('df',str(i),': ',df_before.shape,end = ',')
        except:
            df_before = pd.DataFrame() #����ũ�Ѹ����� ���������� �� df
            print('df',str(i),': ',df_before.shape)
            
        df = pd.concat([df,df_before]) #��ġ��
            

            
    print('�����ģdf : ', df.shape)
    df.drop_duplicates(subset = None,keep = 'first', inplace = True,ignore_index = True)
    print('�ߺ�����df : ', df.shape)#��ü���� ���� �ߺ�����
    #����, Ű,������ ��� null�ƴϰ� ��ġ���� ��� �ϳ� ���� �ִ� �͵� filter
    df = df[~df['gender'].isnull()&~df['height'].isnull()&~df['weight'].isnull()&
                                      (~df['����'].isnull()|
                                       ~df['�㸮�ܸ�'].isnull()|~df['������ܸ�'].isnull()|
                                       ~df['����'].isnull()|
                                       ~df['�شܴܸ�'].isnull()|
                                       ~df['�����̴ܸ�'].isnull())]
    print('������� data:{}'.format(df.shape))
    print('-'*10)
    return df


# %%
# %%

def crawlingdataprocessing(df):
    #kg,cm���� �� ��ġŸ������ ����
    df["height"]=df["height"].replace('cm','',regex = True)
    df["weight"]=df["weight"].replace('kg','',regex = True)
    df= df.astype({'height':'float','weight':'float'})
    print('df��ó���Ϸ�')
    print('-'*10)
    return df

def add_reviewcol(df):
    df = df.drop_duplicates(subset=['content'])#�����ߺ�����
    print('�������븮�� �ߺ����� �Ϸ�')
    df.reset_index(drop=True, inplace=True)

    df['review'] = str('')
    #�ܱ��� ���� ����
    i = 0; nohangul = []
    for i in range(df.shape[0]):
        text = re.sub('[^��-�Ӱ�-�R]', '',df.iloc[i,8])
        if(text==''):
            nohangul.append(i)
    df = df.iloc[[True if i not in nohangul else False for i in range(df.shape[0])],:]
    df.reset_index(drop=True, inplace=True)

    i=0
    for i in range(df.shape[0]):
        text = df.iloc[i,7]
        text = re.sub(pattern='[^\w\s\n]', repl='', string=text) #Ư������ ����
        text = re.sub(pattern='[^��-����-�Ӱ�-�Ra-zA-Z]', repl=' ', string=text) #����, ���̿� ����
        text = re.sub(pattern='[��-����-��]+', repl='', string=text) #�ܼ� ����, ���� ����
        text = repeat_normalize(text, num_repeats=2) #���ʿ�ݺ���������ȭ
        #text = spacing(text) #����------------------------>�����!
        df['review'][i] = text
    print('���� ��ó�� �� �߰�')
    print('df shape:{}'.format(df.shape))
    print('-'*10)
    return df
# %%
# test = pd.read_csv('data/denim_page1.csv')
# test.info()
df = denim_crawlingdataconcat()
df = crawlingdataprocessing(df)
df_before_nlp = add_reviewcol(df)
df_before_nlp_origin = df_before_nlp.copy()
# %%
df_before_nlp.to_csv('data/denim������������_��ó����.csv',encoding = 'UTF-8',index=False)
# %%
