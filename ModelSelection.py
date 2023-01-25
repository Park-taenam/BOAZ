# %%
import numpy as np
import pandas as pd
import joblib
import pickle
from pycaret.regression import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")
# %%
def preprocessing(df):
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

    # Drop Null value
    df_chongjang_big.dropna(axis = 0,inplace = True)
    df_chongjang_small.dropna(axis = 0,inplace = True)
    df_chongjang_soso.dropna(axis = 0,inplace = True)

    df_arm_big.dropna(axis = 0,inplace = True)
    df_arm_small.dropna(axis = 0,inplace = True)
    df_arm_soso.dropna(axis = 0,inplace = True)

    df_chest_big.dropna(axis = 0,inplace = True)
    df_chest_small.dropna(axis = 0,inplace = True)
    df_chest_soso.dropna(axis = 0,inplace = True)

    df_shoulder_big.dropna(axis = 0,inplace = True)
    df_shoulder_small.dropna(axis = 0,inplace = True)
    df_shoulder_soso.dropna(axis = 0,inplace = True)
    
    # train df list
    hood_chongjang_train_lst = [df_chongjang_big, df_chongjang_small, df_chongjang_soso]
    hood_shoulder_train_lst = [df_shoulder_big, df_shoulder_small, df_shoulder_soso]
    hood_chest_train_lst = [df_chest_big, df_chest_small, df_chest_soso]
    hood_arm_train_lst = [df_arm_big, df_arm_small, df_arm_soso]
    
    return hood_chongjang_train_lst, hood_shoulder_train_lst, hood_chest_train_lst, hood_arm_train_lst

def pycaret(hood_chongjang_train_lst, hood_shoulder_train_lst, hood_chest_train_lst, hood_arm_train_lst):
    # 총장
    for i in range(3):
        print("hood_chongjang_train_lst[{}]".format(i))
        s = setup(hood_chongjang_train_lst[i], target = '총장')
        set_config('seed', 2023)
        best = compare_models()
        plot_model(best)
        evaluate_model(best)
    
    # 어깨
    for i in range(3):
        print("hood_shoulder_train_lst[{}]".format(i))
        s = setup(hood_shoulder_train_lst[i], target = '어깨너비')
        set_config('seed', 2023)
        best = compare_models()
        plot_model(best)
        evaluate_model(best)
    
    # 가슴
    for i in range(3):
        print("hood_chest_train_lst[{}]".format(i))
        s = setup(hood_chest_train_lst[i], target = '가슴단면')
        set_config('seed', 2023)
        best = compare_models()
        plot_model(best)
        evaluate_model(best)
    
    # 소매
    for i in range(3):
        print("hood_arm_train_lst[{}]".format(i))
        s = setup(hood_arm_train_lst[i], target = '소매길이')
        set_config('seed', 2023)
        best = compare_models()
        plot_model(best)
        evaluate_model(best)

def lr_trainModel(lst, sizetype):
    for i in range(len(lst)):

        X = lst[i].iloc[:, :-1]
        y = lst[i].iloc[:, -1]

        xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

        lr = LinearRegression()
        lr.fit(xTrain, yTrain)
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

    print("------------------", sizetype, "lr_trainModel:done")

def gbr_trainModel(lst, sizetype):
    for i in range(len(lst)):

        X = lst[i].iloc[:, :-1]
        y = lst[i].iloc[:, -1]

        xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

        gbr = GradientBoostingRegressor(random_state=42)

        parameters = {'learning_rate': [0.001, 0.01, 0.05, 0.1],
                      'subsample': [0.9, 0.5, 0.2, 0.1],
                      'n_estimators': [100, 200, 300, 500],
                      'max_depth': [2, 4, 6, 8]}

        grid_gbr = GridSearchCV(estimator=gbr,
                                param_grid=parameters,
                                cv=3,
                                n_jobs=-1)
        grid_gbr.fit(xTrain, yTrain)
        print("Best estimator", grid_gbr.best_estimator_)

        model = grid_gbr.best_estimator_
        model.fit(xTrain, yTrain)
        prediction = model.predict(xTest)
        print('score:{0:.4f}'.format(model.score(xTest, yTest)))

        if i % 3 == 1:
            print("MSE of small {}'s Model : {}".format(lst[i].columns[-1], mean_squared_error(yTest, prediction)))
            joblib.dump(model, 'model/'+sizetype+'_small_gbrModel.pkl')
        elif i % 3 == 2:
            print("MSE of soso {}'s Model : {}".format(lst[i].columns[-1], mean_squared_error(yTest, prediction)))
            joblib.dump(model, 'model/'+sizetype+'_soso_gbrModel.pkl')
        elif i % 3 == 0:
            print("MSE of big {}'s Model : {}".format(lst[i].columns[-1], mean_squared_error(yTest, prediction)))
            joblib.dump(model, 'model/'+sizetype+'_big_gbrModel.pkl')

    print("------------------", sizetype, "gbr_trainModel:done")

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
if __name__=='__main__':
    df = pd.read_pickle('data_review_preprocessing_done.pkl')

    hood_chongjang_train_lst, hood_shoulder_train_lst, hood_chest_train_lst, hood_arm_train_lst = preprocessing(df)

    # Pycaret
    pycaret(hood_chongjang_train_lst, hood_shoulder_train_lst, hood_chest_train_lst, hood_arm_train_lst)

    ## Train model
    # Linear Regressor
    lr_trainModel(hood_chongjang_train_lst, 'chongjang')
    lr_trainModel(hood_shoulder_train_lst, 'shoulder')
    lr_trainModel(hood_chest_train_lst, 'chest')
    lr_trainModel(hood_arm_train_lst, 'arm')
    # Gradient Boosting Regressor
    gbr_trainModel(hood_chongjang_train_lst, 'chongjang')
    gbr_trainModel(hood_shoulder_train_lst, 'shoulder')
    gbr_trainModel(hood_chest_train_lst, 'chest')
    gbr_trainModel(hood_arm_train_lst, 'arm')

    # Decide Weight
    chongjang_weight = decide_weight(hood_chongjang_train_lst, 'chongjang')
    chest_weight = decide_weight(hood_chest_train_lst, 'chest')
    shoulder_weight = decide_weight(hood_shoulder_train_lst, 'shoulder')
    arm_weight = decide_weight(hood_arm_train_lst, 'arm')
    
    with open('weight.pkl', 'wb') as f:
        pickle.dump([chongjang_weight, chest_weight, shoulder_weight, arm_weight], f)