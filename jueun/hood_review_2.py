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

from konlpy.tag import Kkma #ÇüÅÂ¼ÒºÐ¼®
from konlpy.tag import Okt #ÇüÅÂ¼ÒºÐ¼®
import soynlp; from soynlp.normalizer import * #Á¤±ÔÈ­
# from pykospacing import Spacing------------------>½á¾ßÇÔ!
# spacing = Spacing()------------------------------>½á¾ßÇÔ!

from konlpy.tag import Komoran, Hannanum, Kkma, Okt
komoran = Komoran(); hannanum = Hannanum(); kkma = Kkma(); okt = Okt()
# %%
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
        text = re.sub(pattern='[^¤¡-¤¾¤¿-¤Ó°¡-ÆRa-zA-Z]', repl=' ', string=text) #¼ýÀÚ, ±×ÀÌ¿Ü »èÁ¦
        text = re.sub(pattern='[¤¡-¤¾¤¿-¤Ó]+', repl='', string=text) #´Ü¼ø ¸ðÀ½, ÀÚÀ½ »èÁ¦
        text = repeat_normalize(text, num_repeats=2) #ºÒÇÊ¿ä¹Ýº¹¹®ÀÚÁ¤±ÔÈ­
        #text = spacing(text) #¶ç¾î¾²±â------------------------>½á¾ßÇÔ!
        df['review'][i] = text
    print('¸®ºä ÀüÃ³¸® ¿­ Ãß°¡')
    print('df shape:{}'.format(df.shape))
    print('-'*10)
    return df
# %%

# %%
# df.to_csv('data/hood_before_add_review.csv',index = False,encoding = 'UTF-8')
# df = pd.read_csv('data/hood_before_add_review.csv')
# df.head(2)
df = pd.read_pickle('data/crawlingdata_preprocess_done.pkl')
df_before_nlp = add_reviewcol(df)
df_before_nlp_origin = df_before_nlp.copy()

# %%
# µÎ°³ÀÇ »çÀüÀÌ ÇÕÄ¡°í ³ª¼­ ÁøÇà
# Áßº¹Á¦°Å
# 'n+2'°³ÀÇ ÃÖÁ¾ ¸®½ºÆ®°¡ »ý¼ºµÊ
def sizedict_nodup(oneOfRawList):
    print('Áßº¹Á¦°Å Àü:{}°³'.format(len(oneOfRawList)))
    oneOfRawList = set(oneOfRawList)
    print('Áßº¹Á¦°Å ÈÄ:{}°³'.format(len(oneOfRawList)))
    print('-'*10)
    return list(oneOfRawList)
# %%
total = ['»çÀÌÁî','»çÀÌÁîÇÏ','»çÀÌÁî°¨','ÀÓ»çÀÌÁî','¿©»çÀÌÁî','»çÀÌÁð','»µ¿ä»çÀÌÁî','½ÎÀÌÁî','ÇÍÀÌ','ÇÍ','ÇÍÀÔ´Ï´Ù',
      'ÇÍ°¨µµ','ÇÍ³ª','ÇÍ°¨ÀÌ¶û','ÇÍÃ³·³','ÇÍÀÔ´Ï´ç','ÇÍÀÌ¿¡¿ä','ÇÍÀÌ¿¹¿ä','ÇÍÀÌ¶û','ÇÍÀÌ±¸¿ä','ÇÍÀÌ³×¿ä','ÇÍ°¨Àº',
      'ÇÍÀÔ´Ï','ÇÍÀÌ¶ó¼­','ÇÍÀ¸·Î','¿ÊÇÍ','ÇÍ´õ','ÇÍµµ','ÇÍ°¨','¿Ê','ÈÄµå','ÈÄµåÆ¼','ÇÍÀº','ÁÁ¾Æ¿äÇÍ','¿äÇÍµµ',
      'ÁÁ¾Æ¿äÇÍµµ','»óÀÇ','¿äÇÍ','Æû','Æø','»çÀÌÁî','ÈÄµå','¿Ê','Ç°','ÀüÃ¼','Ç°µµ','¸öÅë','¿ÊÇÍ','»óÃ¼','ÇÍ',
      'ÇÍÀÌ±¸','ÇÍÀÌ¾ú','ÇÍÀÓ','ÇÍÀÔ',
      'ÇÍÀÔ´Ï','ÇÍ°¨','¿äÇÍ','Æû','ÈÄµåÇÍ','ÇÍÀÌ','Å©±â','»óÇ°','»óÀÇ','ÇÍ´õ']
chongjang = ['±æÀÌ','±âÀå','ÃÑÀå','±Ãµð','±Ãµù','¾ûµ¢ÀÌ','¹Ø','¹Ø´Ü','À§¾Æ·¡','³¡´Ü','¹Ø±âÀå',
             '¹Ø´ÜÀÌ¶û','±âÀåÀÌ','±âÀåÀÔ´Ï','¹Ø´Ü','¹Ø','±âÀå°¨','¿ä±âÀå','ÃÑ±æÀÌ','ÃÑÀå','ÃÑ±âÀå',
            '±æÀÌ°¨','³¡','±âÀå','±æÀÌ']
shoulder = ['¾î±úÇÍ','¾î²²','¾î±ú','³Êºñ','¾î±ø','°ñ°Ý','¹Ùµð','°¡·Î','³ÐÀÌ','µîÆÇ','Åë',
            '¾î±ú±øÆÐ','¾î±ú¶ó','¾î±ú³Ð','¾î±úÂÊ','¾î²²','¸öÁý','¾î±úÇÍ','¾î±ú¼±','¾î±ú¶óÀÎ','¾î±ø',
            '¾î±ú','Åë','¸öÅë','°ñ°Ý','³ÐÀÌ','³Êºñ','µîÆÇ']

chest = ['°¡½¿','µÑ·¹','¹Ùµð','°¡·Î','³ÐÀÌ','Åë','µÑ·¹','¸öÁý','°¡½¿ÆÅ','Åë','¸öÅë']
arm = ['¼Ò¸Å','ÆÈµµ','¼Õ¸ñ','¼Õ','ÆÈ´Ù¸®','ÆÈ¸ñ','ÆÈ','ÆÈ²ÞÄ¡','ÆÈÀÌ','¿äÆÈ',
       '´çÆÈ','ÆÈ±æÀÌ','ÆÈ´Ù¸®','ÆÈ±æ','ÆÈ¼Ò¸Å','¼Õ¸ñ','ÆÈ¸ñ','¼Ò¸ÅÅë','¼Õ','ÆÈ','¼Ò¸Å','ÆÈ±âÀå']

small = ['ÀÛ½À´Ï´Ù','Âª¾Ò¾î¿ä','Âª½À´Ï´Ù','ÂªÀ¸¸é','Âª','ÀÛ¾Ò¾î¿ä','Âª´Ù°í','ÂÉ¿©¼­','ÀÛ¾ÒÀ¸¸é',
      'ÀÛ¾Æµµ','ÀÛÀ¸¸é','ÀÛÀ½','ÀÛ¾ÆÁø','Á¼´Ù°í',
      'Âª¾Æ¿ä','Âª°Ô','Âª´Ù´Â','Á¼´Ù´Â','ÀÛ³×','ÀÛ±â´Â','´Þ¶óºÙ´Â','ÀÛÀºµ¥','ÂªÀºµ¥','ÂªÀ½',
      'ÀÛ±ä','Âª³×¿ä','ÀÛ°Ô','ÀÛ¾Æ¼­','ÀÛÀº','Âª°í','Âª¾Æ','Á¼Àº','Å¸ÀÌÆ®','Âª¾ÒÁö¸¸','ºÙ¾î¼­','Âª±ä',
      'ÀÛÀº°Í','ÀÛ¾Æ¿ä','ÂªÀº','Âª¾Æ¼­','ÂªÁö¸¸','Á¼¾Æ','Á¼°í','ÀÛ³×¿ä','ÀÛ¾Æ','ÂÉÀÌ´Â','Á¼¾Æ¿ä','Å©·Ó´À³¦',
      'Å©·ÓÇÏ','ÀÛ°í','¼ô','ÂÌ·Á','ÀÛ´Ù°í','ÀÛ´Ù','Á¼¾Æ¼­','ÀÛ','Å©·Ó','Å©·ÓµÈ','¹Ì´Ï','Å©·ÓÇØ',
      '±æ¾úÀ¸¸é','ÄÇÀ¸¸é','Á¼°Ô','ÂªÁö' ,'ÀÛÁö','Å©·ÓÀÌ','ÇÍµÇ',
      '¼ôÇÏ','»çÀÌÁî¾÷ÇÏ','Å©·ÓÇÏ','¼ôÇÑ','Å©·ÓÇÏ±ä','Å©·ÓÀÌ','Å©·ÓÇØ','Å©·Ó´À³¦','Å©·ÓµÈ','»çÀÌÁî¾÷','Â¦',
       'ÀÛ','¼ô','Âª±ä','Å©·Ó','Å©·ÓÇÑ','Å¸ÀÌÆ®','Å©·ÓÀÌ¶ó','ÀÛ±ä',
       'ÀÛ','Âª','Á¼']

big = ['Å«','±ä','Ä¿¿ä','Å©°í','Å©±ä','³Ë³ËÇÏ°í','±æ¾î¼­','³Ë³ËÇÏ°Ô','ÀÛ´Ù´Â'
     '³Ë³ËÇÑ','Å¬','±æ°í','Å«µ¥','Å©Áö','Å©Áö¸¸','ºÎÇØ','Å©´Ï±ñ','Å©´õ¶ó±¸¿ä','ÄÇ¾îµµ','³ÐÀ½','Å­Á÷ÇÏ°Ô',
     '¹Ú½ÃÇÏ³×¿ä','³Ë³ËÇÒ','Çæ··ÇÏ°Ô','Å­Á÷ÇÏ´Ï','ÄÇÀ½','Å®´Ï','±äµ¥','±æ¾î','±æ°Ô','³Ð¾î¼­','³Ð¾î','¹Ú½ÃÇÏ°í','ÄÇ¾î¿ä',
     '¹Ú½Ã','Å©´Ù°í','Å©´Ù´Â','³Ð°Ô','³Ð°í','Ä¿¿ë','±æ´Ù´Â','Å©Áöµµ','Å­','Å©´Ù','±æ¾ú¾î¿ä','³Ë³ËÇÔ','Å«µ¥µµ','µ¤´Âµ¥',
     'µ¤ÀÌ°í','ÄÇ°í','º¡º¡ÇØ¼­','Å©´õ¶ó°í¿ä','ÄÇÀ»','Å©°Å³ª','Å©´õ¶ó','ÄÇ¾ú´Âµ¥','ÄÇ¾î¿ë','ÄÇ½À´Ï´Ù','µ¤³×¿ä',
     'Å­Áö¸·ÇØ¼­','´þ´Â','¹Ú½ÃÇÏ°Ô','Å©³×¿ä','Å¬±î','³Ë³ËÇÏ±¸','¹Ú½ÃÇÏ','ÄÇ´Âµ¥','³Ë³ËÇÏ³×¿ä','Å©°Ô','Å©´Ï±î','³Ë³ËÇÑµ¥','±æ¾î¿ä',
     'ÄÇÁö¸¸','Çæ··ÇÑ','Å©³×','³Ë³ËÇØ','ÆãÆÛÁüÇÑ','µ¤¾î¿ë','¹Ú½ÃÇÏ´Ï','Çæ··ÇØ','³Ë³ËÇÏÁö''³Ë³ËÇÏ±¸¿ä','Ä¿','Å®´Ï´Ù','³Ë³ËÇÏ´Ï',
     'Å©','±æÁö','¹Ú½ÃÇÑµ¥','µ¤¾î¿ä','±æ±ä','³Ë³ËÇØ¿ä','³ÐÀº','µ¤È÷´Â', '³Ð±ä','µ¤¾îÁà¼­','µ¤Àº','³Ë³ËÇØ¼­','Å­Á÷ÇØ¼­','±â³×','³«³«',
     'Èê·¯³»¸®´Â','µ¤¾î¼­','º¡º¡ÇÑ','Ä¿¼­','Å«°Í','¹Ú½ÃÇÍ','¿ÀÆÛÇÍ','³«³«ÇØ','¿©À¯ÇÍ','³Î³ÎÇÔ','±æ','µ¤¾î','³Ë³ËÇÕ´Ï´Ù','Å©¸é',
     'µ¤°í','°¡¿À¸®','·çÁîÇÍ','ÆãÆÛÁü','±æ¾úÁö¸¸','Çæ··ÇÏ°í','³²¾Æ','¾Æ¹æÇÍ','¹Ú½ºÇÍ','³Î³Î','¿À¹ÙÇÍ','µ¤½À´Ï´Ù','³Ë³ËÈ÷',
     'Çæ··ÇØ¼­','Á¢¾îµµ','Á¢¾î','Á¢¾î¼­','³Î','³Ð¾î¿ä','µ¤°íµµ','¿À¹ö','¿ÍÀÌµå','³¶³¶','Çæ··','Á¢À¸¸é','Âª¾ÒÀ¸¸é','Á¢°í',
     '¹öÇÍ','Á¢¾î¾ß','Àâ¾Æ¸ÔÈù','¼ö¼±ÇØ¼­','¿À¹öÇÍ','¿À¹ö»çÀÌÁî','¿À¹ö»çÀÌÂ¡ÇØ¼­','³«³«ÇÏ´Ù','³«³«ÇÑµ¥','³«³«ÇØ¿ä','¿À¹öÇÍÀÌ¿¹¿ä','¿À¹öÇÍÀÔ´Ï´ç',
       '¿ä¹Ú½ÃÇÏ','ÄÇ¾î¿ë','³«³«ÇÕ´Ï','¾îº¡º¡ÇÏ','¿ÍÀÌµåÇÍ','Å©´Ï±ñ','Å©³×¿ä¿Ê','³Î³ÎÇØ¿ä','¿À¹öÇÍ³ª¿Í¿ä','³Î·²ÇÏ',
       '¿ä³Ë³ËÇÏ','¾öÃ»Å­','¿©À¯ÇÍ','¿À¹ö´À³¦','¾Æ¹æÇÍ','¿À¹öÇÍÀº','³Ð±ä','³Ë³ËÇÔ','ÀÔ´Ï´Ù¿À¹öÇÍ','Å®´Ï´Ù¿À¹öÇÍ',
       'Å®´Ï´ÙÁ¦','Å®´Ï´ç','¿À¹öÇØ¿ä','¿ä¹Ú½ÃÇÑ','Çæ··ÇÏ±ä','Çæ··°Å¸®','¾Æ¹æ¹æÇÑ','¾Æ¹æ¹æ','¹Ú½ÃÇÕ´Ï','¹Ú½ÃÇß´Ù',
       'ºò»çÀÌÁî','¹Þ¾Ò½À´Ï´Ù¿À¹öÇÍ','ºý½ÃÇÏ','·çÁîÇÍÀ¸·Î','³¶³¶ÇÑ','¿À¹öÇÍÀÌ±ä','¹Ú½ºÇÍ','³Î³ÎÇÔ','¾îº¡º¡',
       '¿ä³Ë³ËÇÑ','¿ÀÆÛÇÍ','¿À¹öÇÍÀÌ±¸¿ä','³«³«ÇØ','³¶³¶ÇØ¼­','³Ë³ËÇÏ±¸','³Ë³ËÇÑµ¥','¾Æ¹æÇØ','¿À¹öÇÍÀÌ¶ó¼­',
       '³¶³¶ÇÏ','¹Ú½ÃÇØ','Ä¿¿ë','³Ë³ËÇØ','ÆãÆÛÁü','¹Ú½ÃÇÏÁö','³Î³Î','ÁÁ¾Æ¿ä¿À¹öÇÍ','¾îº¡º¡ÇÑ','³Î³ÎÇÏ','¿À¹ÙÇÍ',
       '¿À¹öÇÏÁö','¿À¹öÇÍÇÏ','¹Ú½ÃÇÕ´Ï´Ù','¿À¹öÇÍµÇ','¹Ú½ÃÇÔ','¹öÇÍ','¹Ú½ÃÇÑ','³«³«ÇÏ´Ï','¹Ú½ÃÇØ¿ä','Å©³×¿ë',                     '³«³«ÇØ¼­','Å­Á÷ÇØ¼­','·çÁîÇØ','´À½¼','³Î³ÎÇÕ´Ï´Ù','¾Æ¹æ¾Æ','¾Æ¹æ¾Æ¹æ','·çÁîÇÍÀÌ','¹Ú½ÃÇÏ','¼¼¹Ì¿À¹ö',
       '¼¼¹Ì¿À¹öÇÍ','·çÁîÇØ¼­','¿À¹öÇÍÀÌ¶ó','·çÁîÇÍÀÌ¶ó','¿À¹öÇÍÀÌ³×¿ä','¿À¹öÇÍÀÔ´Ï','¾îº¡','¿À¹öÇØ','¹Ú½ÃÇÑµ¥',
       '¾Æ¹æ','¿À¹öÇÍÀÌ¿¡¿ä','¿¹»µ¿ä¿À¹öÇÍ','Å­Áö¸·','¹Ú½ÃÇÍ','¿À¹öÇØ¼­','·çÁîÇÑ','¿À¹öÇÍÀÎµ¥','³Ë³ËÇÏ±¸¿ä','¹Ú½ÃÇÒ',
       'Å­Á÷','¿À¹öÇÑ','¿À¹öÇÍÀÌ','³«³«ÇÏ','¾Æ¹æÇÏ','·Õ','¿À¹öÇÍÀ¸·Î','³«³«ÇÑ','¿À¹öÇÍÀÔ´Ï´Ù','¹Ú½ÃÇØ¼­','¿ÍÀÌµå',                     '¿ä¿À¹öÇÍ','³Ë³ËÇØ¼­','Å®´Ï','¿À¹ö','¿À¹öÇÏ','·çÁî','·çÁîÇÍ','¹Ú½Ã','³Ë³Ë','±æ','Å©±ä','¾Æ¹æÇÑ','³Î³ÎÇÑ',
       'Çæ··','±æ±ä','¿ÍÀÌµåÇÏ','¾Æ¹æÇØ¼­','¼¼¹Ì¿À¹ö»çÀÌÁî',             
       'Å©','±æ','³Î','µ¤','³Ð','º¡º¡ÇÏ','µ¤ÀÌ','µ¤È÷','Ä¿´Ù¶þ','´þÈ÷','³Ë³ËÇØÁö','µÚÁý'             
       '³Ë³Ë']
# %%
total     = sizedict_nodup(total)
chongjang = sizedict_nodup(chongjang)
shoulder  = sizedict_nodup(shoulder)
arm       = sizedict_nodup(arm)
chest     = sizedict_nodup(chest)
small     = sizedict_nodup(small)
big       = sizedict_nodup(big)
# %%
#total,chongjang,shoulder,chest,arm ->keywords¸®½ºÆ®

from tqdm import tqdm
def get_keywords(keywords, df,keyword_column_name):
    start = time.time()
    review = df['review']
    df['new_column'] = str('')
    for i in tqdm(range(len(review))):
        # if i % 1000 ==0:
        #     print("{}¹øÂ° ¸®ºä ¿Ï·á(Ãµ ´ÜÀ§) : ".format(i))
        keywords_search = []
        for j in keywords:
            if re.findall(j, review[i]):
                a = re.findall(j +'+[¤¡-¤¾|¤¿-¤Ó|°¡-ÆR]+\s+[¤¡-¤¾|¤¿-¤Ó|°¡-ÆR]+\s+[¤¡-¤¾|¤¿-¤Ó|°¡-ÆR]+',review[i]) #Å°¿öµå +´Ü¾î
                aa = re.findall(j + ' '+'+[¤¡-¤¾|¤¿-¤Ó|°¡-ÆR]+\s+[¤¡-¤¾|¤¿-¤Ó|°¡-ÆR]+\s+[¤¡-¤¾|¤¿-¤Ó|°¡-ÆR]+',review[i])#Å°¿öµå +¶ç°í ´Ü¾î
                b = re.findall('[¤¡-¤¾|¤¿-¤Ó|°¡-ÆR]+' + j ,review[i]) #´Ü¾î + Å°¿öµå
                bb = re.findall('[¤¡-¤¾|¤¿-¤Ó|°¡-ÆR]+\s+' + j ,review[i])#´Ü¾î + ¶ç°í Å°¿öµå
                #bb = re.findall('[¤¡-¤¾|¤¿-¤Ó|°¡-ÆR]+\s+' +' '+ j ,review[i])
                
                  #¾Õ¿¡ Áö¸¸/µ¥/°í/º¸´Ù µîÀÌ µé¾îÀÖÀ¸¸é Å°¿öµå¿¡ ´ëÇÑ ¼ö½Ä¾î±¸°¡ ¾Æ´Ô! 
                if(len(b)!=0):
                    if(('Áö¸¸' in b[0])|('µ¥' in b[0])|('°í' in b[0])|('º¸´Ù' in b[0])|
                       ('µµ' in b[0])|('¼­' in b[0])|('¿ä' in b[0])):
                        b=[]
                    else:
                        pass
                else:
                    pass
                if(len(bb)!=0):
                    if(('Áö¸¸' in bb[0])|('µ¥' in bb[0])|('°í' in bb[0])|('º¸´Ù' in bb[0])|
                       ('µµ' in bb[0])|('¼­' in bb[0])|('¿ä' in bb[0])):
                        bb=[]
                    else:
                        pass
                    
                keywords_search.extend(a)
                keywords_search.extend(aa)
                keywords_search.extend(b)
                keywords_search.extend(bb)
                # print(i,'¹øÂ°',j,a,end='|')
                # print(aa,end='|')
                # print(b,end='|')
                # print(bb)
                
                
        #print(i,'¹øÂ° ¸®½ºÆ®:',keywords_search)
        #print('-'*10)
        
        
        if len(keywords_search) != 0:
            keywords_o = ','.join(x for x in keywords_search)
            df['new_column'][i] = keywords_o 
            
        else:
            df['new_column'][i] = '0'
    
    a = df.rename(columns = {'new_column':keyword_column_name})
    end = time.time()
    print("{:.5f} sec".format(end-start))
    return a[keyword_column_name]
# %%
#total,chongjang,shoulder,chest,arm ->keywords¸®½ºÆ®
keyword_column_name = 'total_keyword'
df_total            = get_keywords(total,df_before_nlp,keyword_column_name)
keyword_column_name = 'chongjang_keyword'
df_chongjang        = get_keywords(chongjang,df_before_nlp,keyword_column_name)
keyword_column_name = 'shoulder_keyword'
df_shoulder         = get_keywords(shoulder,df_before_nlp,keyword_column_name)
keyword_column_name = 'chest_keyword'
df_chest            = get_keywords(chest,df_before_nlp,keyword_column_name)
keyword_column_name = 'arm_keyword'
df_arm              = get_keywords(arm,df_before_nlp,keyword_column_name)

df_keywords         = pd.concat([df_before_nlp,
                                 df_total,
                                df_chongjang,
                                df_chest,
                                df_shoulder,
                                df_arm],axis = 1)
df_keywords =df_keywords.drop(['new_column'],axis = 1)
df_keywords_origin =df_keywords.copy()
# %%
#df_keywords.info()
# %%
#small,big °¨Áö
def big_small(big,small,df,name_big_small):
    start = time.time()
    review = df[name_big_small+'_keyword']
    df[name_big_small + '_big'] = str('')
    df[name_big_small + '_small'] = str('')
    for i in tqdm(range(len(review))):
        # if i % 1000 ==0:
        #     print("{}¹øÂ° ¸®ºä ¿Ï·á(Ãµ ´ÜÀ§) : ".format(i))
        big_search = []
        small_search = []
        for j in big:
            if re.findall(j, review[i]):
                a = re.findall(j,review[i]) #Å°¿öµå
                big_search.extend(a)
               
        for j in small:
            if re.findall(j, review[i]):
                a = re.findall(j,review[i]) #Å°¿öµå
                small_search.extend(a)
            
               
        
        if len(big_search) != 0:
            df[name_big_small + '_big'][i] = 1
            
        else:
            df[name_big_small + '_big'][i] = 0
        
        if len(small_search) != 0:
            df[name_big_small + '_small'][i] = 1
            
        else:
            df[name_big_small + '_small'][i] = 0
        #time.sleep(0.2)
    
    #df.rename(columns = {'keyword_column_name':keyword_column_name},inplace=True)
    end = time.time()
    print("{:.5f} sec".format(end-start))
    return df[[name_big_small + '_big',name_big_small + '_small']]
# %%
name_big_small      = 'total'
total_big_small     = big_small(big,small,df_keywords,name_big_small)
name_big_small      = 'chongjang'
chongjang_big_small = big_small(big,small,df_keywords,name_big_small)
name_big_small      = 'arm'
arm_big_small       = big_small(big,small,df_keywords,name_big_small)
name_big_small      = 'chest'
chest_big_small     = big_small(big,small,df_keywords,name_big_small)
name_big_small      = 'shoulder'
shoulder_big_small  = big_small(big,small,df_keywords,name_big_small)

final_df            = df_keywords
final_df_origin     = final_df.copy()
# %%
final_df.to_pickle('data/crawlingdata_preprocess_review_done.pkl')
# %%
