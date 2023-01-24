# %%
import numpy as np
import pandas as pd
import re
import soynlp
from soynlp.normalizer import *
from pykospacing import Spacing

import warnings
warnings.filterwarnings('ignore')
# %%
def RoadConcat():
    df = pd.DataFrame()
    for page in range(1,21,1):
        df_before = pd.read_csv('./hood_data/hood_{}page.csv'.format(page))
        print("hood_df_{}'s shape : {}".format(page, df_before.shape))
            
        df = pd.concat([df,df_before])
        
    print("\nConcat_df's shape : ", df.shape)

    return df

def Preprocessing(df):
    # 중복제거
    df.drop_duplicates(subset = None,
                       keep = 'first',
                       inplace = True,
                       ignore_index = True)
    print("DropDuplicate_df's shape : {}".format(df.shape))
    
    # NULL 처리
    user_info_notnull_idx = (df['gender'].notnull() & df['height'].notnull() & df['weight'].notnull())
    size_info_notnull_idx = (df['총장'].notnull() | df['어깨너비'].notnull() | df['가슴단면'].notnull() | df['소매길이'].notnull())
    df = df.loc[user_info_notnull_idx&size_info_notnull_idx ,:]
    print("DropNull_df's shape : {}".format(df.shape))

    # 전처리
    df['gender'] = df['gender'].apply(lambda x:x.replace("남성","1") if x == "남성" else x) # 남성 -> 1
    df['gender'] = df['gender'].apply(lambda x:x.replace("여성","0") if x == "여성" else x) # 여성 -> 0
    
    df["height"] = df["height"].replace('cm','',regex = True) # 'cm' 제거
    df["weight"] = df["weight"].replace('kg','',regex = True) # 'kg' 제거
    
    df = df.astype({'gender':'int',
                   'height':'float',
                   'weight':'float'})
    print("Final_df's shape : {}".format(df.shape))
    return df

def ReviewPreprocessing(df):
    # 리뷰중복제거
    df.drop_duplicates(subset=['content'], inplace=True) 
    df.reset_index(drop=True, inplace=True)
    print("리뷰중복제거 \ndf's shape : {}".format(df.shape))
    
    # 외국어 리뷰 삭제
    nohangul = []
    for i in range(df.shape[0]):
        text = re.sub('[^ㄱ-ㅣ가-힣]', '', df.loc[i, 'content'])
        if(text==''):
            nohangul.append(i)
    df = df.iloc[[True if i not in nohangul else False for i in range(df.shape[0])],:]
    df.reset_index(drop=True, inplace=True)
    
    print("외국어 리뷰 삭제 \ndf's shape : {}".format(df.shape))
    
    # review column 생성
    df['review'] = str('')
    spacing = Spacing()
    for i in range(df.shape[0]):
        text = df.loc[i,'content']
        text = re.sub(pattern='[^\w\s\n]', repl='', string=text) # 특수문자 제거
        text = re.sub(pattern='[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z]', repl=' ', string=text) # 숫자, 그 이외 삭제
        text = re.sub(pattern='[ㄱ-ㅎㅏ-ㅣ]+', repl='', string=text) # 단순 모음, 자음 삭제
        text = repeat_normalize(text, num_repeats=2) # 불필요반복문자정규화
        text = spacing(text) # 띄어쓰기
        df.loc[i, 'review'] = text
    print('리뷰 전처리 열 추가 \ndf shape:{}'.format(df.shape))
    
    return df   

# %%
if __name__=="__main__":
    # Data Road,Concat,Preprocessing,ReviewPreprocessing
    data = RoadConcat()
    data = Preprocessing(data)
    data = ReviewPreprocessing(data)

    data.to_pickle('data_preprocessing_done.pkl')