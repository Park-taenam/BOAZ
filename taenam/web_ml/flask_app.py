# %%
from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
import dataframe_image as dfi
import os

from Recsys import *
# %%

app = Flask(__name__)

dir_containing_file = r'c:\Users\User\Desktop\BOAZ\Adv\Project_SiZoAH\taenam\web_ml'
# dir_containing_file = r'/home/SizeRecSysTest2/mysite'
os.chdir(dir_containing_file)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("Size_RecSys.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    # Road Weight
    with open('weight.pkl', 'rb') as f:
        chongjang_weight, chest_weight, shoulder_weight, arm_weight = pickle.load(f)

    if request.form['Gender'] in ['남성', '남', "Male"]:
        gender = 1 
    elif request.form['Gender'] in ['여성', '여', "Female"]:
        gender = 0
        
    if request.form['Chongjang'] == "큼":
        userChongjangPrefer = 1
    elif request.form['Chongjang'] == "작음":
        userChongjangPrefer = -1
    elif request.form['Chongjang'] == "보통":
        userChongjangPrefer = 0
    
    if request.form['Shoulder'] == "큼":
        userShoulderPrefer = 1
    elif request.form['Shoulder'] == "작음":
        userShoulderPrefer = -1
    elif request.form['Shoulder'] == "보통":
        userShoulderPrefer = 0

    if request.form['Chest'] == "큼":
        userChestPrefer = 1
    elif request.form['Chest'] == "작음":
        userChestPrefer = -1
    elif request.form['Chest'] == "보통":
        userChestPrefer = 0

    if request.form['Arm'] == "큼":
        userArmPrefer = 1
    elif request.form['Arm'] == "작음":
        userArmPrefer = -1
    elif request.form['Arm'] == "보통":
        userArmPrefer = 0

    ## Recsys
    userHeight = float(request.form['Height'])
    userWeight = float(request.form['Weight'])
    userGender = int(gender)
    
    # 사용자 수치 예측
    user_prediction_lst = finalSizeRecSys(userHeight, userWeight, userGender,
                                          userChongjangPrefer, userShoulderPrefer, userChestPrefer, userArmPrefer,
                                          chongjang_weight, shoulder_weight, chest_weight, arm_weight)
    # 최종 사이즈 추천
    data = [[62, 50, 59, 60], 
            [67, 56, 63, 64],
            [70, 59, 66, 67]]
    hood_size_df = pd.DataFrame(data, index=["S","M","L"], columns=["총장", "어깨너비", "가슴단면", "소매길이"])
    
    size_rec = find_mse_in_size_df(hood_size_df, user_prediction_lst)
    
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        
        return array[idx]
    def df_coloring_length(series):
        highlight = 'background-color : FFD700;'
        default = ''

        nearest_value = find_nearest(series, user_prediction_lst[0])

        return [highlight if e == nearest_value else default for e in series]
    def df_coloring_shoulder(series):
        highlight = 'background-color : FFD700;'
        default = ''
        
        nearest_value = find_nearest(series, user_prediction_lst[1])
        
        return [highlight if e == nearest_value else default for e in series]
    def df_coloring_bl(series):
        highlight = 'background-color : FFD700;'
        default = ''
        
        nearest_value = find_nearest(series, user_prediction_lst[2])
        
        return [highlight if e == nearest_value else default for e in series]
    def df_coloring_sleeve(series):
        highlight = 'background-color : FFD700;'
        default = ''
        
        nearest_value = find_nearest(series, user_prediction_lst[3])
        
        return [highlight if e == nearest_value else default for e in series]
    
    hood_size_df = hood_size_df.style.apply(df_coloring_length, subset=["총장"], axis=0).apply(df_coloring_shoulder, subset=["어깨너비"], axis=0).apply(df_coloring_bl, subset=["가슴단면"], axis=0).apply(df_coloring_sleeve, subset=["소매길이"], axis=0)
    dfi.export(hood_size_df, './static/image/size_df_image.png', max_cols=-1, max_rows=-1)
    
    return render_template('Size_RecSys.html',
                           user_info= '키/몸무게/성별 : {} / {} / {}'.format(userHeight, userWeight, request.form['Gender']),
                           prefer_ch = '총장 : {}'.format(request.form['Chongjang']),
                           prefer_sh = '어깨너비 : {}'.format(request.form['Shoulder']),
                           prefer_chest = '가슴단면 : {}'.format(request.form['Chest']),
                           prefer_ar = '소매길이 : {}'.format(request.form['Arm']),
                           pred='최종 추천 사이즈 : {}'.format(size_rec),
                           image_file='image/size_df_image.png')
    
if __name__ == '__main__':
    app.run(debug=True)
