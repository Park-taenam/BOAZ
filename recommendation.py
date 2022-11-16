'''
Content : Recommendation system - collaborative filtering
Data : 20221114
Author : Taenam
env : sizoah/pycaret_env
Reference
    pycaret install : https://github.com/pycaret/pycaret/issues/1260
    pycaret example : https://pycaret.gitbook.io/docs/learn-pycaret/examples
    pickle error : https://optilog.tistory.com/34
'''

# %% Import
# from nltk.corpus import stopwords
# import datetime
# from tqdm import tqdm_notebook, tqdm   # for문 진행상황 눈으로 확인 (loading bar)
# from PIL import Image
# from collections import Counter
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# from nltk.tokenize import word_tokenize
# import nltk
# from konlpy.tag import *   # 모든 형태소분석기 import 하기
# import seaborn as sns
# import pandas as pd
# import numpy as np
# import os
# import sys
# import gc
# import re
# import io
# import matplotlib.pyplot as plt

# from sklearn.metrics.pairwise import cosine_similarity
# import pycaret.regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import EDA

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# %% Road and preprocessing data
data_top1 = pd.read_csv('./data/mutandard_top1.csv', encoding='cp949', index_col=0)
data_top1 = EDA.preprocessing(data_top1)

# %% KNN + KNeighborsClassifier
# ## KNN Classifier
# X = data_top1.loc[:, ['height', 'weight']].values
# y = data_top1.loc[:, 'size'].values

# scaler_x = StandardScaler()
# scaler_x.fit(X)
# X_scaled = scaler_x.transform(X)

# # plt.scatter(pd.DataFrame(X)[0], pd.DataFrame(X)[1])
# # plt.show()
# # plt.scatter(pd.DataFrame(X_scaled)[0], pd.DataFrame(X_scaled)[1])
# # plt.show()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# knn = KNeighborsClassifier(n_neighbors=5)  # 하이퍼파라미터 조정 필요
# knn.fit(X_train, y_train)

# y_pred = knn.predict(X_test)

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# %% New process
## 수치 추가
## https://www.musinsa.com/app/goods/1216295
# outseam : 총장, waist : 허리단면, thigh : 허벅지 단면, rise : 밑위, hem : 밑단단면
data_top1_size_lst = [[26, 92, 35, 27.3, 23.5, 16.5],
                     [27, 92, 36.3, 27.9, 23.5, 16.8],
                     [28, 93, 37.5, 28.5, 24, 17],
                     [29, 93, 38.8, 29.1, 24, 17.3],
                     [30, 94, 40, 29.8, 24.5, 17.5],
                     [31, 94, 41.3, 30.4, 24.5, 17.8],
                     [32, 94, 42.5, 31, 25, 18],
                     [33, 94, 43.8, 31.6, 25.5, 18.3],
                     [34, 95, 45, 32.3, 26, 18.5],
                     [36, 95, 47.5, 33.5, 27, 19],
                     [38, 96, 50, 34.8, 28, 19.5],
                     [40, 96, 52.5, 36, 29, 20],
                     [42, 97, 55, 37.3, 30, 20.5]]
data_top1_size_df = pd.DataFrame(data_top1_size_lst, 
                                 columns = ['size', 'outseam', 'waist', 'thigh', 'rise', 'hem'])
data_top1 = pd.merge(data_top1, data_top1_size_df, on='size', how='inner')

# gender imbalance여서 우선 제외
data = data_top1.loc[:, ['height', 'weight', 'size_eval', 'outseam', 'waist', 'thigh', 'rise', 'hem']]

data_outseam = data.loc[:, ['height', 'weight', 'size_eval', 'outseam']]
data_waist = data.loc[:, ['height', 'weight', 'size_eval', 'waist']]
data_thigh = data.loc[:, ['height', 'weight', 'size_eval', 'thigh']]
data_rise = data.loc[:, ['height', 'weight', 'size_eval', 'rise']]
data_hem = data.loc[:, ['height', 'weight', 'size_eval', 'hem']]

data_list = [data_outseam, data_waist, data_thigh, data_rise, data_rise]

# %% pycaret
'''
pycaret : 우선 보류
'''
# ## pycaret
# demo = pycaret.regression.setup(data = data_list[0], target = 'outseam', 
#                                 # ignore_features = [],
#                                 # normalize = True,
#                                 # transformation= True,
#                                 # transformation_method = 'yeo-johnson',
#                                 # transform_target = True,
#                                 # remove_outliers= True,
#                                 # remove_multicollinearity = True,
#                                 # ignore_low_variance = True,
#                                 # combine_rare_levels = True
#                                 ) 

# best = pycaret.regression.compare_models()
# # plot_model(best)
# # evaluate_model(best)

# ## Creating models for the best estimators
# random_forest = pycaret.regression.create_model('rf')

# # ## Tuning the created models 
# # random_forest = pycaret.tune_model(random_forest)

# ## Finaliszing model for predictions 
# test_data = data_outseam.iloc[:10, :]
# predictions = pycaret.regression.predict_model(random_forest, data = test_data)

# %% Phase 1
'''
pycaret에서 결과 가장 좋았던 random forest 사용
'''
X_outseam = data_list[0].iloc[:, :-1]
y_outseam = data_list[0].iloc[:, -1]

xTrain, xTest, yTrain, yTest = train_test_split(X_outseam, y_outseam, test_size = 0.3, random_state = 531)

mseOos = []
nTreeList = range(50, 500, 10)

for iTrees in tqdm(nTreeList, desc='iterate list'):
    depth = None
    RFModel = RandomForestRegressor(n_estimators=iTrees,
                                    max_depth=depth,
                                    oob_score=False, 
                                    random_state=531)
    RFModel.fit(xTrain, yTrain)
    
    #데이터 세트에 대한 MSE 누적
    prediction = RFModel.predict(xTest)
    mseOos.append(mean_squared_error(yTest, prediction))

# MSE visualization
plt.plot(nTreeList, mseOos)
plt.xlabel('Number of Trees in Ensemble')
plt.ylabel('Mean Squared Error')
#plot.ylim([0.0, 1.1*max(mseOob)])
plt.show()

regr = RandomForestRegressor(n_estimators = nTreeList[np.argmin(mseOos)],
                             random_state=531)
regr.fit(xTrain, yTrain)
prediction = regr.predict(xTest)
print(mean_squared_error(yTest, prediction))

# %%
userTest = [[178, 76, 0]]
prediction_user = regr.predict(userTest)


# %%
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def df_coloring(value):
    highlight = 'background-color : darkorange;'
    default = ''

    if find_nearest(data_top1_size_df.loc[:, 'outseam'], prediction_user) == value:
        return highlight
    return default

data_top1_size_df.style.applymap(df_coloring)