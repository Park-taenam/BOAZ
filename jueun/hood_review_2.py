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
komoran = Komoran(); hannanum = Hannanum(); kkma = Kkma(); okt = Okt()
# %%
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

# %%

df = pd.read_pickle('data/crawlingdata_preprocess_done.pkl')
df_before_nlp = add_reviewcol(df)
df_before_nlp_origin = df_before_nlp.copy()
# %%
# �ΰ��� ������ ��ġ�� ���� ����
# �ߺ�����
# 'n+2'���� ���� ����Ʈ�� ������
def sizedict_nodup(oneOfRawList):
    print('�ߺ����� ��:{}��'.format(len(oneOfRawList)))
    oneOfRawList = set(oneOfRawList)
    print('�ߺ����� ��:{}��'.format(len(oneOfRawList)))
    print('-'*10)
    return list(oneOfRawList)
# %%
total = ['������','��������','�����','�ӻ�����','��������','������','���������','������','����','��','���Դϴ�',
      '�Ͱ���','�ͳ�','�Ͱ��̶�','��ó��','���Դϴ�','���̿���','���̿���','���̶�','���̱���','���̳׿�','�Ͱ���',
      '���Դ�','���̶�','������','����','�ʹ�','�͵�','�Ͱ�','��','�ĵ�','�ĵ�Ƽ','����','���ƿ���','���͵�',
      '���ƿ��͵�','����','����','��','��','������','�ĵ�','��','ǰ','��ü','ǰ��','����','����','��ü','��',
      '���̱�','���̾�','����','����',
      '���Դ�','�Ͱ�','����','��','�ĵ���','����','ũ��','��ǰ','����','�ʹ�']
chongjang = ['����','����','����','�õ�','�õ�','������','��','�ش�','���Ʒ�','����','�ر���',
             '�ش��̶�','������','�����Դ�','�ش�','��','���尨','�����','�ѱ���','����','�ѱ���',
            '���̰�','��','����','����']
shoulder = ['�����','�','���','�ʺ�','���','���','�ٵ�','����','����','����','��',
            '�������','�����','�����','�����','�','����','�����','�����','�������','���',
            '���','��','����','���','����','�ʺ�','����']

chest = ['����','�ѷ�','�ٵ�','����','����','��','�ѷ�','����','������','��','����']
arm = ['�Ҹ�','�ȵ�','�ո�','��','�ȴٸ�','�ȸ�','��','�Ȳ�ġ','����','����',
       '����','�ȱ���','�ȴٸ�','�ȱ�','�ȼҸ�','�ո�','�ȸ�','�Ҹ���','��','��','�Ҹ�','�ȱ���']

small = ['�۽��ϴ�','ª�Ҿ��','ª���ϴ�','ª����','ª','�۾Ҿ��','ª�ٰ�','�ɿ���','�۾�����',
      '�۾Ƶ�','������','����','�۾���','���ٰ�',
      'ª�ƿ�','ª��','ª�ٴ�','���ٴ�','�۳�','�۱��','�޶�ٴ�','������','ª����','ª��',
      '�۱�','ª�׿�','�۰�','�۾Ƽ�','����','ª��','ª��','����','Ÿ��Ʈ','ª������','�پ','ª��',
      '������','�۾ƿ�','ª��','ª�Ƽ�','ª����','����','����','�۳׿�','�۾�','���̴�','���ƿ�','ũ�Ӵ���',
      'ũ����','�۰�','��','�̷�','�۴ٰ�','�۴�','���Ƽ�','��','ũ��','ũ�ӵ�','�̴�','ũ����',
      '�������','������','����','ª��' ,'����','ũ����','�͵�',
      '����','���������','ũ����','����','ũ���ϱ�','ũ����','ũ����','ũ�Ӵ���','ũ�ӵ�','�������','¦',
       '��','��','ª��','ũ��','ũ����','Ÿ��Ʈ','ũ���̶�','�۱�',
       '��','ª','��']

big = ['ū','��','Ŀ��','ũ��','ũ��','�˳��ϰ�','��','�˳��ϰ�','�۴ٴ�'
     '�˳���','Ŭ','���','ū��','ũ��','ũ����','����','ũ�ϱ�','ũ���󱸿�','�Ǿ','����','ŭ���ϰ�',
     '�ڽ��ϳ׿�','�˳���','�混�ϰ�','ŭ���ϴ�','����','Ů��','�䵥','���','���','�о','�о�','�ڽ��ϰ�','�Ǿ��',
     '�ڽ�','ũ�ٰ�','ũ�ٴ�','�а�','�а�','Ŀ��','��ٴ�','ũ����','ŭ','ũ��','������','�˳���','ū����','���µ�',
     '���̰�','�ǰ�','�����ؼ�','ũ�������','����','ũ�ų�','ũ����','�Ǿ��µ�','�Ǿ��','�ǽ��ϴ�','���׿�',
     'ŭ�����ؼ�','����','�ڽ��ϰ�','ũ�׿�','Ŭ��','�˳��ϱ�','�ڽ���','�Ǵµ�','�˳��ϳ׿�','ũ��','ũ�ϱ�','�˳��ѵ�','����',
     '������','�混��','ũ��','�˳���','��������','�����','�ڽ��ϴ�','�混��','�˳�����''�˳��ϱ���','Ŀ','Ů�ϴ�','�˳��ϴ�',
     'ũ','����','�ڽ��ѵ�','�����','���','�˳��ؿ�','����','������', '�б�','�����༭','����','�˳��ؼ�','ŭ���ؼ�','���','����',
     '�귯������','���','������','Ŀ��','ū��','�ڽ���','������','������','������','�γ���','��','����','�˳��մϴ�','ũ��',
     '����','������','������','������','�������','�混�ϰ�','����','�ƹ���','�ڽ���','�γ�','������','�����ϴ�','�˳���',
     '�混�ؼ�','���','����','���','��','�о��','������','����','���̵�','����','�混','������','ª������','����',
     '����','�����','��Ƹ���','�����ؼ�','������','����������','��������¡�ؼ�','�����ϴ�','�����ѵ�','�����ؿ�','�������̿���','�������Դϴ�',
       '��ڽ���','�Ǿ��','�����մ�','�����','���̵���','ũ�ϱ�','ũ�׿��','�γ��ؿ�','�����ͳ��Ϳ�','�η���',
       '��˳���','��ûŭ','������','��������','�ƹ���','��������','�б�','�˳���','�Դϴٿ�����','Ů�ϴٿ�����',
       'Ů�ϴ���','Ů�ϴ�','�����ؿ�','��ڽ���','�混�ϱ�','�混�Ÿ�','�ƹ����','�ƹ��','�ڽ��մ�','�ڽ��ߴ�',
       '�������','�޾ҽ��ϴٿ�����','������','����������','������','�������̱�','�ڽ���','�γ���','���',
       '��˳���','������','�������̱���','������','�����ؼ�','�˳��ϱ�','�˳��ѵ�','�ƹ���','�������̶�',
       '������','�ڽ���','Ŀ��','�˳���','������','�ڽ�����','�γ�','���ƿ������','�����','�γ���','������',
       '��������','��������','�ڽ��մϴ�','�����͵�','�ڽ���','����','�ڽ���','�����ϴ�','�ڽ��ؿ�','ũ�׿�',                     '�����ؼ�','ŭ���ؼ�','������','����','�γ��մϴ�','�ƹ��','�ƹ�ƹ�','��������','�ڽ���','���̿���',
       '���̿�����','�����ؼ�','�������̶�','�������̶�','�������̳׿�','�������Դ�','�','������','�ڽ��ѵ�',
       '�ƹ�','�������̿���','�����������','ŭ����','�ڽ���','�����ؼ�','������','�������ε�','�˳��ϱ���','�ڽ���',
       'ŭ��','������','��������','������','�ƹ���','��','����������','������','�������Դϴ�','�ڽ��ؼ�','���̵�',                     '�������','�˳��ؼ�','Ů��','����','������','����','������','�ڽ�','�˳�','��','ũ��','�ƹ���','�γ���',
       '�混','���','���̵���','�ƹ��ؼ�','���̿���������',             
       'ũ','��','��','��','��','������','����','����','Ŀ�ٶ�','����','�˳�����','����'             
       '�˳�']
# %%
total     = sizedict_nodup(total)
chongjang = sizedict_nodup(chongjang)
shoulder  = sizedict_nodup(shoulder)
arm       = sizedict_nodup(arm)
chest     = sizedict_nodup(chest)
small     = sizedict_nodup(small)
big       = sizedict_nodup(big)
# %%
#total,chongjang,shoulder,chest,arm ->keywords����Ʈ

from tqdm import tqdm
def get_keywords(keywords, df,keyword_column_name):
    start = time.time()
    review = df['review']
    df['new_column'] = str('')
    for i in tqdm(range(len(review))):
        
        keywords_search = []
        for j in keywords:
            if re.findall(j, review[i]):
                a = re.findall(j +'+[��-��|��-��|��-�R]+\s+[��-��|��-��|��-�R]+\s+[��-��|��-��|��-�R]+',review[i]) #Ű���� +�ܾ�
                aa = re.findall(j + ' '+'+[��-��|��-��|��-�R]+\s+[��-��|��-��|��-�R]+\s+[��-��|��-��|��-�R]+',review[i])#Ű���� +��� �ܾ�
                b = re.findall('[��-��|��-��|��-�R]+' + j ,review[i]) #�ܾ� + Ű����
                bb = re.findall('[��-��|��-��|��-�R]+\s+' + j ,review[i])#�ܾ� + ��� Ű����
                
                
                  #�տ� ����/��/��/���� ���� ��������� Ű���忡 ���� ���ľ�� �ƴ�! 
                if(len(b)!=0):
                    if(('����' in b[0])|('��' in b[0])|('��' in b[0])|('����' in b[0])|
                       ('��' in b[0])|('��' in b[0])|('��' in b[0])):
                        b=[]
                    else:
                        pass
                else:
                    pass
                if(len(bb)!=0):
                    if(('����' in bb[0])|('��' in bb[0])|('��' in bb[0])|('����' in bb[0])|
                       ('��' in bb[0])|('��' in bb[0])|('��' in bb[0])):
                        bb=[]
                    else:
                        pass
                    
                keywords_search.extend(a)
                keywords_search.extend(aa)
                keywords_search.extend(b)
                keywords_search.extend(bb)

        
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
#total,chongjang,shoulder,chest,arm ->keywords����Ʈ
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
#small,big ����
def big_small(big,small,df,name_big_small):
    start = time.time()
    review = df[name_big_small+'_keyword']
    df[name_big_small + '_big'] = str('')
    df[name_big_small + '_small'] = str('')
    for i in tqdm(range(len(review))):
      
        big_search = []
        small_search = []
        for j in big:
            if re.findall(j, review[i]):
                a = re.findall(j,review[i]) #Ű����
                big_search.extend(a)
               
        for j in small:
            if re.findall(j, review[i]):
                a = re.findall(j,review[i]) #Ű����
                small_search.extend(a)
            
               
        
        if len(big_search) != 0:
            df[name_big_small + '_big'][i] = 1
            
        else:
            df[name_big_small + '_big'][i] = 0
        
        if len(small_search) != 0:
            df[name_big_small + '_small'][i] = 1
            
        else:
            df[name_big_small + '_small'][i] = 0
        
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
# 11ó���ϰ� �̰Ŵ� ������ �ڵ����ľ���, �״����� �������ϴ� �ڵ� �����ϰ� 
# ����ġ �ڵ������ϰ�  �����ؼ� ������� �ڵ� ����!

# %%
