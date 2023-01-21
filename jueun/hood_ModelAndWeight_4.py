# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#import EDA
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score
# %%
def forModel_preprocessing(df):
    df = df.astype({'chongjang_big':'int',
                'chongjang_small':'int',
                'shoulder_big':'int',
                'shoulder_small':'int',
                'chest_big':'int',
                'chest_small':'int',
                'arm_big':'int',
                'arm_small':'int'})
    df = df.loc[:, ['gender', 'height', 'weight', 'size', 'content',
                    '총장', '어깨너비', '가슴단면', '소매길이',
                    'chongjang_big', 'chongjang_small',
                    'shoulder_big', 'shoulder_small',
                    'chest_big', 'chest_small',
                    'arm_big', 'arm_small']]
    
    # Make train set
    df_chongjang_big    = df.loc[df['chongjang_big'] == 1, ['height','weight','gender', '총장']]
    df_chongjang_small  = df.loc[df['chongjang_small'] == 1, ['height','weight','gender', '총장']]
    df_chongjang_soso   = df.loc[(df['chongjang_big'] == 0) & (df['chongjang_small'] == 0), ['height','weight','gender', '총장']]

    df_shoulder_big     = df.loc[df['shoulder_big'] == 1, ['height','weight','gender', '어깨너비']]
    df_shoulder_small   = df.loc[df['shoulder_small'] == 1, ['height','weight','gender', '어깨너비']]
    df_shoulder_soso    = df.loc[(df['shoulder_big'] == 0) & (df['shoulder_small'] == 0), ['height','weight','gender', '어깨너비']]

    df_chest_big        = df.loc[df['chest_big'] == 1, ['height','weight','gender', '가슴단면']]
    df_chest_small      = df.loc[df['chest_small'] == 1, ['height','weight','gender', '가슴단면']]
    df_chest_soso       = df.loc[(df['chest_big'] == 0 )& (df['chest_small'] == 0), ['height','weight','gender', '가슴단면']]

    df_arm_big          = df.loc[df['arm_big'] == 1, ['height','weight','gender','소매길이']]
    df_arm_small        = df.loc[df['arm_small'] == 1, ['height','weight','gender','소매길이']]
    df_arm_soso         = df.loc[(df['arm_big'] == 0) & (df['arm_small'] == 0), ['height','weight','gender','소매길이']]
    
    # train df list
    hood_chongjang_train_lst = [df_chongjang_big, df_chongjang_small, df_chongjang_soso]
    hood_shoulder_train_lst = [df_shoulder_big, df_shoulder_small, df_shoulder_soso]
    hood_chest_train_lst = [df_chest_big, df_chest_small, df_chest_soso]
    hood_arm_train_lst = [df_arm_big, df_arm_small, df_arm_soso]
    
    return hood_chongjang_train_lst, hood_shoulder_train_lst, hood_chest_train_lst, hood_arm_train_lst
# %%
df = pd.read_pickle('data/Modeling_DF_230116.pickle'); df = df.dropna(axis=0) # ---------->다정이에게 크다작다모두 11인것 처리끝난 파일!! 3이후의 파일로 고쳐야하고 두번째 모든 null값없애는 것은 1전처리파일에 추가완료
hood_chongjang_train_lst, hood_shoulder_train_lst, hood_chest_train_lst, hood_arm_train_lst = forModel_preprocessing(df)
# %%
# linear regression 모델 저장하는 것 
# #위치는 model파일에 저장됨
def lr_trainModel(lst,sizetype):
    for i in range(len(lst)):
        
        X = lst[i].iloc[:, :-1]
        y = lst[i].iloc[:, -1]

        xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 42)
        #print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape)
        
        lr = LinearRegression()
        lr.fit(xTrain,yTrain)
        prediction = lr.predict(xTest)
        
        if i % 3 == 1:
            print("MSE of small {}'s Model : {}".format(lst[i].columns[-1], mean_squared_error(yTest, prediction)))
            joblib.dump(lr, 'model/'+sizetype+'_small_lrModel.pkl')
        elif i % 3 == 2:
            print("MSE of soso {}'s Model : {}".format(lst[i].columns[-1], mean_squared_error(yTest, prediction)))
            joblib.dump(lr, 'model/'+sizetype+'_soso_lrModel.pkl')
        elif i % 3 == 0:
            print("MSE of big {}'s Model : {}".format(lst[i].columns[-1], mean_squared_error(yTest, prediction)))
            joblib.dump(lr, 'model/'+sizetype+'_big_lrModel.pkl')

    print("------------------",sizetype,"lr_trainModel:done")
        
# gradient boosting regressor 모델 저장하는 것 
# #위치는 model파일에 저장됨
def gbr_trainModel(lst,sizetype):
    for i in range(len(lst)):
        
        X = lst[i].iloc[:, :-1]
        y = lst[i].iloc[:, -1]

        xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 42)
        #print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape)
        
        
        gbr = GradientBoostingRegressor()

        parameters = {'learning_rate': [0.01,0.05,0.1,0.15],
                  'subsample'    : [0.9, 0.5, 0.2, 0.1],
                  'n_estimators' : [100,200,300,500],
                  'max_depth'    : [4,6,8,10]
                 }

        grid_gbr = GridSearchCV(estimator=gbr, param_grid = parameters, cv = 2, n_jobs=-1)
        grid_gbr.fit(xTrain,yTrain)
        print("Best estimator",grid_gbr.best_estimator_)
        
        model = grid_gbr.best_estimator_
        model.fit(xTrain,yTrain)
        prediction = model.predict(xTest)
        print('score:{0:.4f}'.format(model.score(xTest,yTest)))
        
        if i % 3 == 1:
            print("MSE of small {}'s Model : {}".format(lst[i].columns[-1], mean_squared_error(yTest, prediction)))
            joblib.dump(model, 'model/'+sizetype+'_small_gbrModel.pkl')
        elif i % 3 == 2:
            print("MSE of soso {}'s Model : {}".format(lst[i].columns[-1], mean_squared_error(yTest, prediction)))
            joblib.dump(model, 'model/'+sizetype+'_soso_gbrModel.pkl')
        elif i % 3 == 0:
            print("MSE of big {}'s Model : {}".format(lst[i].columns[-1], mean_squared_error(yTest, prediction)))
            joblib.dump(model, 'model/'+sizetype+'_big_gbrModel.pkl')
        
    print("------------------",sizetype,"gbr_trainModel:done")
    # %%
sizetype = 'chongjang'
lr_trainModel(hood_chongjang_train_lst,sizetype)
sizetype = 'shoulder'
lr_trainModel(hood_shoulder_train_lst,sizetype)
sizetype = 'chest'
lr_trainModel(hood_chest_train_lst,sizetype)
sizetype = 'arm'
lr_trainModel(hood_arm_train_lst,sizetype)
# %%
sizetype = 'chongjang'
gbr_trainModel(hood_chongjang_train_lst,sizetype)
sizetype = 'shoulder'
gbr_trainModel(hood_shoulder_train_lst,sizetype)
sizetype = 'chest'
gbr_trainModel(hood_chest_train_lst,sizetype)
sizetype = 'arm'
gbr_trainModel(hood_arm_train_lst,sizetype)
# %%
# 저장된 모델 불러와서 가중치 구하기!! 
def decide_weight(lst,size_type):
    print('-------'+size_type+'-------')
    weight = []
    for i in range(len(lst)):
        
        X = lst[i].iloc[:, :-1]
        y = lst[i].iloc[:, -1]
        xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 42)
        

        load_model_big = joblib.load('model/'+size_type+'_big_lrModel.pkl') # big->lr
        prediction_big = load_model_big.predict(xTest)#a
                
        load_model_small = joblib.load('model/'+size_type+'_small_lrModel.pkl') # small->lr
        prediction_small = load_model_small.predict(xTest)#b
                
        load_model_soso = joblib.load('model/'+size_type+'_soso_gbrModel.pkl') # soso->gbr
        prediction_soso = load_model_soso.predict(xTest)#c

        mse_lst = []
        abc = []

        for a in np.arange(0, 1.0, 0.01):
            for b in np.arange(0, 1 - a, 0.01):
                
                c = 1 - (a + b)
                final_prediction = prediction_big * a + prediction_small* b + prediction_soso * c
                abc.append([a,b,c])
                mse_lst.append(mean_squared_error(yTest, final_prediction))
                
        if i % 3 == 0:
            print("big 일 때! 최소 MSE:{},가중치(big,small,soso):{}".format(mse_lst[np.argmin(mse_lst)],[round(x,2) for x in abc[np.argmin(mse_lst)]]))
            weight.append(abc[np.argmin(mse_lst)])
            
        elif i % 3 == 1:
            print("small 일 때! 최소 MSE:{},가중치(big,small,soso):{}".format(mse_lst[np.argmin(mse_lst)],[round(x,2) for x in abc[np.argmin(mse_lst)]]))
            weight.append(abc[np.argmin(mse_lst)])
            
        elif i % 3 == 2:
            print("soso 일 때! 최소 MSE:{},가중치(big,small,soso):{}".format(mse_lst[np.argmin(mse_lst)],[round(x,2) for x in abc[np.argmin(mse_lst)]]))
            weight.append(abc[np.argmin(mse_lst)])
    
    return weight
# %%             
size_type           = 'chongjang'
chongjang_weight    = decide_weight(hood_chongjang_train_lst,size_type) # '크다, 작다, 중간이다' 가중치 순으로 들어가 있음
size_type           = 'chest'
chest_weight        = decide_weight(hood_chest_train_lst,size_type)
size_type           = 'shoulder'
shoulder_weight     = decide_weight(hood_shoulder_train_lst,size_type)
size_type           = 'arm'
arm_weight          = decide_weight(hood_arm_train_lst,size_type)         

# %%
#최종 4가지 수치 도출
def finalSizeRecSys():
    userHeight= float(input("키:"))
    userWeight = float(input("몸무게:"))
    userGender = int(input("여자는 0, 남자는 1:"))
    userInfo = [[userHeight,userWeight,userGender]]
    print('사용자정보(키,몸무게,성별):',userInfo)
    
    #userTotalPrefer    = int(input("전체적인 핏의 선호를 숫자로 입력하세요. [크다(1),작다(-1),보통이다(0)]:"))
    userChongjangPrefer = int(input("총장 선호를 숫자로 입력하세요. [크다(1),작다(-1),보통이다(0)]:"))
    userShoulderPrefer  = int(input("어깨너비 선호를 숫자로 입력하세요. [크다(1),작다(-1),보통이다(0)]:"))
    userChestPrefer     = int(input("가슴단면 선호를 숫자로 입력하세요. [크다(1),작다(-1),보통이다(0)]:"))
    userArmPrefer       = int(input("소매길이 선호를 숫자로 입력하세요. [크다(1),작다(-1),보통이다(0)]:"))
   
    chongjang_model_big     = joblib.load('model/chongjang_big_lrModel.pkl')
    chongjang_model_small   = joblib.load('model/chongjang_small_lrModel.pkl')
    chongjang_model_soso    = joblib.load('model/chongjang_soso_gbrModel.pkl')
    
    shoulder_model_big      = joblib.load('model/shoulder_big_lrModel.pkl')
    shoulder_model_small    = joblib.load('model/shoulder_small_lrModel.pkl')
    shoulder_model_soso     = joblib.load('model/shoulder_soso_gbrModel.pkl')
    
    chest_model_big         = joblib.load('model/chest_big_lrModel.pkl')
    chest_model_small       = joblib.load('model/chest_small_lrModel.pkl')
    chest_model_soso        = joblib.load('model/chest_soso_gbrModel.pkl')
    
    arm_model_big           = joblib.load('model/arm_big_lrModel.pkl')
    arm_model_small         = joblib.load('model/arm_small_lrModel.pkl')
    arm_model_soso          = joblib.load('model/arm_soso_gbrModel.pkl')
    
    
    #총장예측
    if userChongjangPrefer == 1:    # big선호할 때
        a = chongjang_weight[0][0]
        b = chongjang_weight[0][1]
        c = chongjang_weight[0][2]
    elif userChongjangPrefer == -1: # small 선호할 때
        a = chongjang_weight[1][0]
        b = chongjang_weight[1][1]
        c = chongjang_weight[1][2]
    else:                           # soso 선호할때
        a = chongjang_weight[2][0]
        b = chongjang_weight[2][1]
        c = chongjang_weight[2][2]
    userChonjangPrediction = (chongjang_model_big.predict(userInfo) * a 
                              + chongjang_model_small.predict(userInfo) * b 
                              + chongjang_model_soso.predict(userInfo) * c)
    print('총장 예측값:{}'.format(userChonjangPrediction))
    
    
    if userShoulderPrefer == 1:    # big선호할 때
        a = shoulder_weight[0][0]
        b = shoulder_weight[0][1]
        c = shoulder_weight[0][2]
    elif userShoulderPrefer == -1: # small 선호할 때
        a = shoulder_weight[1][0]
        b = shoulder_weight[1][1]
        c = shoulder_weight[1][2]
    else:                           # soso 선호할때
        a = shoulder_weight[2][0]
        b = shoulder_weight[2][1]
        c = shoulder_weight[2][2]
    userShoulderPrediction = (shoulder_model_big.predict(userInfo) * a 
                              + shoulder_model_small.predict(userInfo) * b 
                              + shoulder_model_soso.predict(userInfo) * c)
    print('어깨너비 예측값:{}'.format(userShoulderPrediction))
        
    if userChestPrefer == 1:    # big선호할 때
        a = chest_weight[0][0]
        b = chest_weight[0][1]
        c = chest_weight[0][2]
    elif userChestPrefer == -1: # small 선호할 때
        a = chest_weight[1][0]
        b = chest_weight[1][1]
        c = chest_weight[1][2]
    else:                       # soso 선호할때
        a = chest_weight[2][0]
        b = chest_weight[2][1]
        c = chest_weight[2][2]
    userChestPrediction = (chest_model_big.predict(userInfo) * a 
                              + chest_model_small.predict(userInfo) * b 
                              + chest_model_soso.predict(userInfo) * c)
    print('가슴단면 예측값:{}'.format(userChestPrediction))
    
    if userArmPrefer == 1:    # big선호할 때
        a = arm_weight[0][0]
        b = arm_weight[0][1]
        c = arm_weight[0][2]
    elif userArmPrefer == -1: # small 선호할 때
        a = arm_weight[1][0]
        b = arm_weight[1][1]
        c = arm_weight[1][2]
    else:                     # soso 선호할때
        a = arm_weight[2][0]
        b = arm_weight[2][1]
        c = arm_weight[2][2]
    userArmPrediction = (arm_model_big.predict(userInfo) * a 
                              + arm_model_small.predict(userInfo) * b 
                              + arm_model_soso.predict(userInfo) * c)
    print('소매길이 예측값:{}'.format(userArmPrediction))
    print('done')
    
    return [userChonjangPrediction, userShoulderPrediction, userChestPrediction, userArmPrediction]
        
# %%
userPrediction_allSize = finalSizeRecSys()
# %%
# 4가지 수치 기반 최종추천사이즈(s,m,l)
hood_size_df = pd.DataFrame([['S', 65, 48, 58, 64],
                            ['M', 67.5, 50, 60.5, 65.5],
                            ['L', 70, 52, 63, 67],
                            ['XL', 72.5, 54, 65.5, 68.5]], columns=["size", "총장", "어깨너비", "가슴단면", "소매길이"])
hood_size_df.set_index('size', inplace=True)
hood_col_dict = {0:"총장", 1:"어깨너비", 2:"가슴단면", 3:"소매길이"}

def find_mse_in_size_df(df, uservalue):
    size_df_mse_lst = []
    size = df.index.to_list()
    for i in range(len(df)):
        mse = mean_squared_error(np.asarray(df.iloc[i,:]), uservalue)
        size_df_mse_lst.append(mse)
    print(size_df_mse_lst)
    return print('최종추천 사이즈:',size[size_df_mse_lst.index(min(size_df_mse_lst))])
# %%
find_mse_in_size_df(hood_size_df, userPrediction_allSize)
# %%
userPrediction_allSize
# %%
'''
아래는 태남씌 df 색칠코드-> 잘돌아갑니다
'''
# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return array[idx]

# def df_coloring_length(series):
#     highlight = 'background-color : blue;'
#     default = ''
    
#     nearest_value = find_nearest(series, userPrediction_allSize[0])
    
#     return [highlight if e == nearest_value else default for e in series]

# def df_coloring_shoulder(series):
#     highlight = 'background-color : blue;'
#     default = ''
    
#     nearest_value = find_nearest(series, userPrediction_allSize[1])
    
#     return [highlight if e == nearest_value else default for e in series]

# def df_coloring_bl(series):
#     highlight = 'background-color : blue;'
#     default = ''
    
#     nearest_value = find_nearest(series, userPrediction_allSize[2])
    
#     return [highlight if e == nearest_value else default for e in series]

# def df_coloring_sleeve(series):
#     highlight = 'background-color : blue;'
#     default = ''
    
#     nearest_value = find_nearest(series, userPrediction_allSize[3])
    
#     return [highlight if e == nearest_value else default for e in series]


# hood_size_df.style.apply(df_coloring_length, subset=["총장"], axis=0).apply(df_coloring_shoulder, subset=["어깨너비"], axis=0).apply(df_coloring_bl, subset=["가슴단면"], axis=0).apply(df_coloring_sleeve, subset=["소매길이"], axis=0)