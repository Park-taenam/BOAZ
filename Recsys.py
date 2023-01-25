# %%
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

# %%
def finalSizeRecSys(userHeight,userWeight,userGender,
                    userChongjangPrefer, userShoulderPrefer, userChestPrefer, userArmPrefer,
                    chongjang_weight, shoulder_weight, chest_weight, arm_weight):
    
    userInfo = [[userHeight,userWeight,userGender]]
    print('사용자정보(키,몸무게,성별):', userInfo)
    
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
    print('총장 예측값:{}'.format(str(userChonjangPrediction)))
    
    
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
    print('어깨너비 예측값:{}'.format(str(userShoulderPrediction)))
        
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
    print('가슴단면 예측값:{}'.format(str(userChestPrediction)))
    
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
    print('소매길이 예측값:{}'.format(str(userArmPrediction)))
    print('done')
    
    user_prediction_lst = [userChonjangPrediction, userShoulderPrediction, userChestPrediction, userArmPrediction]
    
    return user_prediction_lst

def find_mse_in_size_df(size_df, user_prediction_lst):
    size_df_mse_lst = []
    size = size_df.index.to_list()
    for i in range(size_df.shape[0]):
        mse = mean_squared_error(np.asarray(size_df.iloc[i,:]), user_prediction_lst)
        size_df_mse_lst.append(mse)

    return size[np.argmin(size_df_mse_lst)]

# %%
if __name__=='__main__':

    # Road Weight
    with open('./data/weight.pkl', 'rb') as f:
        chongjang_weight, chest_weight, shoulder_weight, arm_weight = pickle.load(f)

    ## Recsys
    userHeight = float(input("키:"))
    userWeight = float(input("몸무게:"))
    userGender = int(input("여자는 0, 남자는 1:"))

    userChongjangPrefer = int(input("총장 선호를 숫자로 입력하세요. [크다(1),작다(-1),보통이다(0)]:"))
    userShoulderPrefer = int(input("어깨너비 선호를 숫자로 입력하세요. [크다(1),작다(-1),보통이다(0)]:"))
    userChestPrefer = int(input("가슴단면 선호를 숫자로 입력하세요. [크다(1),작다(-1),보통이다(0)]:"))
    userArmPrefer = int(input("소매길이 선호를 숫자로 입력하세요. [크다(1),작다(-1),보통이다(0)]:"))
    # 사용자 수치 예측
    user_prediction_lst = finalSizeRecSys(userHeight, userWeight, userGender,
                                          userChongjangPrefer, userShoulderPrefer, userChestPrefer, userArmPrefer,
                                          chongjang_weight, shoulder_weight, chest_weight, arm_weight)
    # 최종 사이즈 추천
    hood_size_df = pd.DataFrame([['S', 65, 48, 58, 64],
                                 ['M', 67.5, 50, 60.5, 65.5],
                                 ['L', 70, 52, 63, 67],
                                 ['XL', 72.5, 54, 65.5, 68.5]], columns=["size", "총장", "어깨너비", "가슴단면", "소매길이"])
    hood_size_df.set_index('size', inplace=True)
    # hood_col_dict = {0:"총장", 1:"어깨너비", 2:"가슴단면", 3:"소매길이"}
    
    size_rec = find_mse_in_size_df(hood_size_df, user_prediction_lst)
    
    print('최종추천 사이즈: {}'.format(size_rec))