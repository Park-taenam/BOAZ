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

# %%
df = pd.DataFrame()
for i in range(1,21,1):
    try:
        df_before = pd.read_csv('data/hood_'+str(i)+'page.csv') #크롤링파일 불러오기
        print('df',str(i),': ',df_before.shape)
    except:
        df_before = pd.DataFrame() #아직크롤링되지 않은파일은 빈 df
        print('df',str(i),': ',df_before.shape)
        
    df = pd.concat([df,df_before]) #합치기
        

        
print('df : ', df.shape)
df.drop_duplicates(subset = None,keep = 'first', inplace = True,ignore_index = True)
print('중복제거df : ', df.shape)
# %%
df.to_csv('data/hood_1to20.csv', encoding="UTF-8", index=False) #파일로 저장
# %%
