'''
Content : EDA, preprocessing
Author : Taenam
'''

# %%
import numpy as np
import pandas as pd
import mlxtend

import warnings
warnings.filterwarnings('ignore')

# %%
def preprocessing(data):
    # height, weight 숫자로 변경
    data['height'] = [int(height.strip().split('c')[0])
                      for height in data['height']]
    data['weight'] = [int(weight.strip().split('k')[0])
                      for weight in data['weight']]

    # 사이즈 평가
    data["size_eval"] = data["size_eval"].replace('보통이에요', '0', regex=True)
    data["size_eval"] = data["size_eval"].replace('커요', '1', regex=True)
    data["size_eval"] = data["size_eval"].replace('작아요', '-1', regex=True)
    data["size_eval"] = pd.to_numeric(data["size_eval"])

    return data


# %%
if __name__=="__main__":
    data_top1 = pd.read_csv('./data/mutandard_top1.csv', encoding='cp949', index_col=0)
    data_top2 = pd.read_csv('./data/mutandard_top2.csv', encoding='cp949', index_col=0)
    
    print("키 종류 수 : ", data_top1['height'].nunique())
    print("몸무게 종류 수 : ", data_top1['weight'].nunique())
    print("상품 종류 수 : ", data_top1['item'].nunique())
    print("사이즈 종류 수 : ", data_top1['size'].nunique())
    print("평점 종류 수 : ", data_top1['star'].nunique())
    print("리뷰 종류 수 : ", data_top1['content'].nunique())
    
    data_top1 = preprocessing(data_top1)
    data_top2 = preprocessing(data_top2)
    
    # create pickle
    data = pd.concat([data_top1, data_top2], axis=0)
    # data = data.reset_index()
    data.to_pickle('./data/data.pkl')