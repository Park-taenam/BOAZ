'''
Content : Recommendation system - collaborative filtering
Data : 221114
Author : Taenam
env : sizoah/pycaret_env
Reference
    pycaret install : https://github.com/pycaret/pycaret/issues/1260
    pycaret example : https://pycaret.gitbook.io/docs/learn-pycaret/examples
    pickle error : https://optilog.tistory.com/34
    - coloring cells in pandas
      * - https://queirozf.com/entries/pandas-dataframe-examples-styling-cells-and-conditional-formatting
    - Dimensionality Reduction with Neighborhood Components Analysis
        - https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_dim_reduction.html
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import EDA

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# %% Road and preprocessing data
# data_top1 = pd.read_csv('./data/mutandard_top1.csv', encoding='cp949', index_col=0)
# data_top1 = EDA.preprocessing(data_top1)

hood_df = pd.read_csv('./data/hoodTop5FigureTest.csv')
hood_df = hood_df.iloc[:, 2:]
hood_df = EDA.preprocessing(hood_df)

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
# data_top1_size_lst = [[26, 92, 35, 27.3, 23.5, 16.5],
#                      [27, 92, 36.3, 27.9, 23.5, 16.8],
#                      [28, 93, 37.5, 28.5, 24, 17],
#                      [29, 93, 38.8, 29.1, 24, 17.3],
#                      [30, 94, 40, 29.8, 24.5, 17.5],
#                      [31, 94, 41.3, 30.4, 24.5, 17.8],
#                      [32, 94, 42.5, 31, 25, 18],
#                      [33, 94, 43.8, 31.6, 25.5, 18.3],
#                      [34, 95, 45, 32.3, 26, 18.5],
#                      [36, 95, 47.5, 33.5, 27, 19],
#                      [38, 96, 50, 34.8, 28, 19.5],
#                      [40, 96, 52.5, 36, 29, 20],
#                      [42, 97, 55, 37.3, 30, 20.5]]
# data_top1_size_df = pd.DataFrame(data_top1_size_lst, 
#                                  columns = ['size', 'outseam', 'waist', 'thigh', 'rise', 'hem'])
# data_top1 = pd.merge(data_top1, data_top1_size_df, on='size', how='inner')

# # gender imbalance여서 우선 제외
# data = data_top1.loc[:, ['height', 'weight', 'size_eval', 'outseam', 'waist', 'thigh', 'rise', 'hem']]

# data_outseam = data.loc[:, ['height', 'weight', 'size_eval', 'outseam']]
# data_waist = data.loc[:, ['height', 'weight', 'size_eval', 'waist']]
# data_thigh = data.loc[:, ['height', 'weight', 'size_eval', 'thigh']]
# data_rise = data.loc[:, ['height', 'weight', 'size_eval', 'rise']]
# data_hem = data.loc[:, ['height', 'weight', 'size_eval', 'hem']]

# data_list = [data_outseam, data_waist, data_thigh, data_rise, data_hem]

hood_df_length = hood_df.loc[:, ['height', 'weight', '총장']]
hood_df_shoulder = hood_df.loc[:, ['height', 'weight', '어깨너비']]
hood_df_bl = hood_df.loc[:, ['height', 'weight', '가슴단면']] # bust line
hood_df_sleeve = hood_df.loc[:, ['height', 'weight', '소매길이']]

hood_df_lst = [hood_df_length, hood_df_shoulder, hood_df_bl, hood_df_sleeve]
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
# ## outseam
# X_outseam = data_list[0].iloc[:, :-1]
# y_outseam = data_list[0].iloc[:, -1]

# xTrain, xTest, yTrain, yTest = train_test_split(X_outseam, y_outseam, test_size = 0.3, random_state = 531)

# mseOos = []
# nTreeList = range(50, 500, 10)

# for iTrees in tqdm(nTreeList, desc='iterate list'):
#     depth = None
#     RFModel = RandomForestRegressor(n_estimators=iTrees,
#                                     max_depth=depth,
#                                     oob_score=False, 
#                                     random_state=531)
#     RFModel.fit(xTrain, yTrain)
    
#     #데이터 세트에 대한 MSE 누적
#     prediction = RFModel.predict(xTest)
#     mseOos.append(mean_squared_error(yTest, prediction))

# # MSE visualization
# plt.plot(nTreeList, mseOos)
# plt.xlabel('Number of Trees in Ensemble')
# plt.ylabel('Mean Squared Error')
# #plot.ylim([0.0, 1.1*max(mseOob)])
# plt.show()

# regr = RandomForestRegressor(n_estimators = nTreeList[np.argmin(mseOos)],
#                              random_state=531)
# regr.fit(xTrain, yTrain)
# prediction = regr.predict(xTest)
# print(mean_squared_error(yTest, prediction))

# userTest = [[178, 76, 0]]
# prediction_user = regr.predict(userTest)

# %%
user_pred_list = []
for i in range(len(hood_df_lst)):
    X = hood_df_lst[i].iloc[:, :-1]
    y = hood_df_lst[i].iloc[:, -1]

    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 531)
    
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
        
    regr = RandomForestRegressor(n_estimators = nTreeList[np.argmin(mseOos)],
                             random_state=531)
    regr.fit(xTrain, yTrain)
    
    prediction = regr.predict(xTest)
    print("MSE of {}'s Model : {}".format(hood_df_lst[i].columns[-1], mean_squared_error(yTest, prediction)))
    
    userTest = [[178, 76]]
    prediction_user = regr.predict(userTest)
    user_pred_list.append(prediction_user)
    print("User Test of {}'s Model : {}".format(hood_df_lst[i].columns[-1], prediction_user))
# %%
'''
원하는 옷의 치수 정보에 가장 가까운 값 표시
'''
## 예시 https://www.musinsa.com/app/goods/2758349
hood_size_df = pd.DataFrame([['S', 65, 48, 58, 64],
                            ['M', 67.5, 50, 60.5, 65.5],
                            ['L', 70, 52, 63, 67],
                            ['XL', 72.5, 54, 65.5, 68.5]], columns=["size", "총장", "어깨너비", "가슴단면", "소매길이"])
# hood_size_df = hood_size_df.iloc[:, 1:]
hood_size_df.set_index('size', inplace=True)
hood_col_dict = {0:"총장", 1:"어깨너비", 2:"가슴단면", 3:"소매길이"}

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def df_coloring_length(series):
    highlight = 'background-color : darkorange;'
    default = ''
    
    nearest_value = find_nearest(series, user_pred_list[0])
    
    return [highlight if e == nearest_value else default for e in series]

def df_coloring_shoulder(series):
    highlight = 'background-color : darkorange;'
    default = ''
    
    nearest_value = find_nearest(series, user_pred_list[1])
    
    return [highlight if e == nearest_value else default for e in series]

def df_coloring_bl(series):
    highlight = 'background-color : darkorange;'
    default = ''
    
    nearest_value = find_nearest(series, user_pred_list[2])
    
    return [highlight if e == nearest_value else default for e in series]

def df_coloring_sleeve(series):
    highlight = 'background-color : darkorange;'
    default = ''
    
    nearest_value = find_nearest(series, user_pred_list[3])
    
    return [highlight if e == nearest_value else default for e in series]


hood_size_df.style.apply(df_coloring_length, subset=["총장"], axis=0).apply(df_coloring_shoulder, subset=["어깨너비"], axis=0).apply(df_coloring_bl, subset=["가슴단면"], axis=0).apply(df_coloring_sleeve, subset=["소매길이"], axis=0)

# %%
'''
1. 각 치수에 해당하는 값들을 예측 후 의류 수치 정보에 가장 가까운 사이즈 추천
    - 그냥 거리로 하면 될 듯 - euclidean distance  
        (차원축소 그 딴거 필요없는 듯)
        (스케일링할지 말지 결정 - test 해보자)
2. 해당 의류를 구매한 사용자들에 가장 가까운 사이즈 추천 -> 이러면 보편화 안됨..!
    - 이건 키/몸무게로 비슷한 범주 선택 (분류 문제)
        KNN
        SVM
    - 치수 정보를 가중치를 부여서 수정한 후 차원축소해서 접근해도 될 듯 
        (지금은 사람마다 치수정보가 동일하므로 의미가 없다)
'''

## 차원축소 (보류)
# random_state = 0

# # Reduce dimension to 2 with PCA
# pca = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=random_state))

# # Reduce dimension to 2 with LinearDiscriminantAnalysis
# lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=2))

# # Reduce dimension to 2 with NeighborhoodComponentAnalysis
# nca = make_pipeline(
#     StandardScaler(),
#     NeighborhoodComponentsAnalysis(n_components=2, random_state=random_state),
# )

# # Make a list of the methods to be compared
# dim_reduction_methods = [("PCA", pca), ("LDA", lda), ("NCA", nca)]

# # Use a nearest neighbor classifier to evaluate the methods
# n_neighbors = 1
# knn = KNeighborsClassifier(n_neighbors=n_neighbors)

# X = hood_size_df.iloc[:, 1:]
# y = hood_size_df.iloc[:, 0]

# for i, (name, model) in enumerate(dim_reduction_methods):
#     plt.figure()
#     # plt.subplot(1, 3, i + 1, aspect=1)

#     # Fit the method's model
#     model.fit(X, y)

#     # Fit a nearest neighbor classifier on the embedded training set
#     knn.fit(model.transform(X), y)

#     # Embed the data set in 2 dimensions using the fitted model
#     X_embedded = model.transform(X)

#     # Plot the projected points and show the evaluation score
#     plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=30, cmap="Set1")
#     plt.title(
#         "{}, KNN (k={})".format(name, n_neighbors)
#     )
# plt.show()

# %%
## 1. 유클리드 거리로 가까운 값 찾기 (스케일링할지말지 고민)

user_pred_int_arr = np.array([float(i) for i in user_pred_list])

dist_lst = []
for idx in tqdm(list(hood_size_df.index)):
    sum_sq = np.sum(np.square(np.array(hood_size_df.loc[idx, :]) - user_pred_int_arr))
    dist = np.sqrt(sum_sq)
    
    dist_lst.append(dist)

print("User info : {}/{}".format(userTest[0][0], userTest[0][1]))
print("User Prediction : {}".format(np.round(user_pred_int_arr,3)))
print("Euclidean distance : {}".format(np.round(dist_lst,3)))
print("추천 사이즈 : {}".format(hood_size_df.index[np.argmin(dist_lst)]))
hood_size_df.style.apply(df_coloring_length, subset=["총장"], axis=0).apply(df_coloring_shoulder, subset=["어깨너비"], axis=0).apply(df_coloring_bl, subset=["가슴단면"], axis=0).apply(df_coloring_sleeve, subset=["소매길이"], axis=0)

# %%
