
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# ½Ã°¢È­ °ü·Ã
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
from PIL import Image
from matplotlib import rc

# ±×·¡ÇÁ¿¡¼­ ÇÑ±Û ÆùÆ® ±úÁö´Â ¹®Á¦¿¡ ´ëÇÑ ´ëÃ³(Àü¿ª ±Û²Ã ¼³Á¤)
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

from konlpy.tag import Kkma #ÇüÅÂ¼ÒºĞ¼®
from konlpy.tag import Okt #ÇüÅÂ¼ÒºĞ¼®
import soynlp; from soynlp.normalizer import * #Á¤±ÔÈ­
# from pykospacing import Spacing------------------>½á¾ßÇÔ!
# spacing = Spacing()------------------------------>½á¾ßÇÔ!

from konlpy.tag import Komoran, Hannanum, Kkma, Okt
komoran = Komoran(); hannanum = Hannanum(); kkma = Kkma(); okt = Okt();
# %%
def denim_crawlingdataconcat():
    df = pd.DataFrame()
    for i in range(1,21,1):
        try:
            df_before = pd.read_csv('data/denim_page'+str(i)+'.csv') #Å©·Ñ¸µÆÄÀÏ ºÒ·¯¿À±â
            print('df',str(i),': ',df_before.shape,end = ',')
        except:
            df_before = pd.DataFrame() #¾ÆÁ÷Å©·Ñ¸µµÇÁö ¾ÊÀºÆÄÀÏÀº ºó df
            print('df',str(i),': ',df_before.shape)
            
        df = pd.concat([df,df_before]) #ÇÕÄ¡±â
            

            
    print('¸ğµÎÇÕÄ£df : ', df.shape)
    df.drop_duplicates(subset = None,keep = 'first', inplace = True,ignore_index = True)
    print('Áßº¹Á¦°Ådf : ', df.shape)#ÀüÃ¼¿­ÀÌ °°Àº Áßº¹Á¦°Å
    #¼ºº°, Å°,¸ö¹«°Ô ¸ğµÎ null¾Æ´Ï°í ¼öÄ¡Á¾·ù Àû¾îµµ ÇÏ³ª °ªÀÌ ÀÖ´Â °Íµé filter
    df = df[~df['gender'].isnull()&~df['height'].isnull()&~df['weight'].isnull()&
                                      (~df['ÃÑÀå'].isnull()|
                                       ~df['Çã¸®´Ü¸é'].isnull()|~df['Çã¹÷Áö´Ü¸é'].isnull()|
                                       ~df['¹ØÀ§'].isnull()|
                                       ~df['¹Ø´Ü´Ü¸é'].isnull()|
                                       ~df['¾ûµ¢ÀÌ´Ü¸é'].isnull())]
    print('ÃÖÁ¾»ç¿ë data:{}'.format(df.shape))
    print('-'*10)
    return df


# %%
# %%

def crawlingdataprocessing(df):
    #kg,cmÁ¦°Å ¹× ¼öÄ¡Å¸ÀÔÀ¸·Î º¯°æ
    df["height"]=df["height"].replace('cm','',regex = True)
    df["weight"]=df["weight"].replace('kg','',regex = True)
    df= df.astype({'height':'float','weight':'float'})
    print('dfÀüÃ³¸®¿Ï·á')
    print('-'*10)
    return df

def add_reviewcol(df):
    df = df.drop_duplicates(subset=['content'])#¸®ºäÁßº¹Á¦°Å
    print('°°Àº³»¿ë¸®ºä Áßº¹Á¦°Å ¿Ï·á')
    df.reset_index(drop=True, inplace=True)

    df['review'] = str('')
    #¿Ü±¹¾î ¸®ºä »èÁ¦
    i = 0; nohangul = []
    for i in range(df.shape[0]):
        text = re.sub('[^¤¡-¤Ó°¡-ÆR]', '',df.iloc[i,8])
        if(text==''):
            nohangul.append(i)
    df = df.iloc[[True if i not in nohangul else False for i in range(df.shape[0])],:]
    df.reset_index(drop=True, inplace=True)

    i=0
    for i in range(df.shape[0]):
        text = df.iloc[i,7]
        text = re.sub(pattern='[^\w\s\n]', repl='', string=text) #Æ¯¼ö¹®ÀÚ Á¦°Å
        text = re.sub(pattern='[^¤¡-¤¾¤¿-¤Ó°¡-ÆRa-zA-Z]', repl=' ', string=text) #¼ıÀÚ, ±×ÀÌ¿Ü »èÁ¦
        text = re.sub(pattern='[¤¡-¤¾¤¿-¤Ó]+', repl='', string=text) #´Ü¼ø ¸ğÀ½, ÀÚÀ½ »èÁ¦
        text = repeat_normalize(text, num_repeats=2) #ºÒÇÊ¿ä¹İº¹¹®ÀÚÁ¤±ÔÈ­
        #text = spacing(text) #¶ç¾î¾²±â------------------------>½á¾ßÇÔ!
        df['review'][i] = text
    print('¸®ºä ÀüÃ³¸® ¿­ Ãß°¡')
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
df_before_nlp.to_csv('data/denim¾ğ¾î»çÀü¸¸µé±â¿ë_ÀüÃ³¸®¿Ï.csv',encoding = 'UTF-8',index=False)
# %%
