#필요한 라이브러리 불러오기
# %%
import pandas as pd;import numpy as np;import re, os, sys, json
from pykospacing import Spacing #띄어쓰기
spacing=Spacing()
from konlpy.tag import Kkma #형태소분석
from konlpy.tag import Okt #형태소분석
import soynlp; from soynlp.normalizer import *
from konlpy.tag import Komoran, Hannanum, Kkma, Okt
komoran = Komoran(); hannanum = Hannanum(); kkma = Kkma(); okt = Okt()
import time
# %%

#%%
#사용할 데이터 불러오기
df = pd.read_csv('data/hood_page1to20_final_nodup.csv', encoding='utf-8')
df.drop_duplicates(subset = None,keep = 'first', inplace = True,ignore_index = True)
print('중복제거df : ', df.shape)#전체열이 같은 중복제거
#성별, 키,몸무게 모두 null아니고 수치종류 적어도 하나 값이 있는 것들 filter
df = df[~df['gender'].isnull()&~df['height'].isnull()&~df['weight'].isnull()&
                                      (~df['총장'].isnull()|
                                       ~df['어깨너비'].isnull()|~df['가슴단면'].isnull()|
                                       ~df['소매길이'].isnull())]
print('최종사용 data:{}'.format(df.shape))
print('-'*10)
df = df.iloc[:,1:]
df = crawlingdataprocessing(df)
df_before_nlp = add_reviewcol(df)
#%%





#----------------필요한 함수 정의------------------
# %%
def crawlingdataconcat():
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
    #성별, 키,몸무게 모두 null아니고 수치종류 적어도 하나 값이 있는 것들 filter
    df = df[~df['gender'].isnull()&~df['height'].isnull()&~df['weight'].isnull()&
                                      (~df['총장'].isnull()|
                                       ~df['어깨너비'].isnull()|~df['가슴단면'].isnull()|
                                       ~df['소매길이'].isnull())]
    print('최종사용 data:{}'.format(df.shape))
    print('-'*10)
    return df

#(2) 크롤링 데이터 기초전처리
def crawlingdataprocessing(df):
    #kg,cm제거 및 수치타입으로 변경
    df["height"]=df["height"].replace('cm','',regex = True)
    df["weight"]=df["weight"].replace('kg','',regex = True)
    df= df.astype({'height':'float','weight':'float'})
    print('df전처리완료')
    print('-'*10)
    return df

#(3) 단순자연어전처리 컬럼 review 생성
def add_reviewcol(df):
    df = df.drop_duplicates(subset=['content'])#리뷰중복제거
    print('같은내용리뷰 중복제거 완료')
    df.reset_index(drop=True, inplace=True)

    df['review'] = str('')
    #외국어 리뷰 삭제
    i = 0; nohangul = []
    for i in range(df.shape[0]):
        text = re.sub('[^ㄱ-ㅣ가-힣]', '',df.iloc[i,8])
        if(text==''):
            nohangul.append(i)
    df = df.iloc[[True if i not in nohangul else False for i in range(df.shape[0])],:]
    df.reset_index(drop=True, inplace=True)

    i=0
    for i in range(df.shape[0]):
        text = df.iloc[i,7]
        text = re.sub(pattern='[^\w\s\n]', repl='', string=text) #특수문자 제거
        text = re.sub(pattern='[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z]', repl=' ', string=text) #숫자, 그이외 삭제
        text = re.sub(pattern='[ㄱ-ㅎㅏ-ㅣ]+', repl='', string=text) #단순 모음, 자음 삭제
        text = repeat_normalize(text, num_repeats=2) #불필요반복문자정규화
        #text = spacing(text) #띄어쓰기
        df['review'][i] = text
    print('리뷰 전처리 열 추가')
    print('df shape:{}'.format(df.shape))
    print('-'*10)
    return df
# %%
