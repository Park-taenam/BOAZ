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

# %%
df = pd.DataFrame()
for i in range(1,21,1):
    try:
        df_before = pd.read_csv('data/hood_'+str(i)+'page.csv') #ũ�Ѹ����� �ҷ�����
        print('df',str(i),': ',df_before.shape)
    except:
        df_before = pd.DataFrame() #����ũ�Ѹ����� ���������� �� df
        print('df',str(i),': ',df_before.shape)
        
    df = pd.concat([df,df_before]) #��ġ��
        

        
print('df : ', df.shape)
df.drop_duplicates(subset = None,keep = 'first', inplace = True,ignore_index = True)
print('�ߺ�����df : ', df.shape)
# %%
df.to_csv('data/hood_1to20.csv', encoding="UTF-8", index=False) #���Ϸ� ����
# %%
