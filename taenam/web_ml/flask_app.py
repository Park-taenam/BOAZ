# %%
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import os
from Recsys import *

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
        
    if request.form['Chongjang'] == "크다":
        userChongjangPrefer = 1
    elif request.form['Chongjang'] == "작다":
        userChongjangPrefer = -1
    elif request.form['Chongjang'] == "보통이다":
        userChongjangPrefer = 0
    
    if request.form['Shoulder'] == "크다":
        userShoulderPrefer = 1
    elif request.form['Shoulder'] == "작다":
        userShoulderPrefer = -1
    elif request.form['Shoulder'] == "보통이다":
        userShoulderPrefer = 0

    if request.form['Chest'] == "크다":
        userChestPrefer = 1
    elif request.form['Chest'] == "작다":
        userChestPrefer = -1
    elif request.form['Chest'] == "보통이다":
        userChestPrefer = 0

    if request.form['Arm'] == "크다":
        userArmPrefer = 1
    elif request.form['Arm'] == "작다":
        userArmPrefer = -1
    elif request.form['Arm'] == "보통이다":
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
    hood_size_df = pd.DataFrame([['S', 65, 48, 58, 64],
                                 ['M', 67.5, 50, 60.5, 65.5],
                                 ['L', 70, 52, 63, 67],
                                 ['XL', 72.5, 54, 65.5, 68.5]], columns=["size", "총장", "어깨너비", "가슴단면", "소매길이"])
    
    hood_size_df.set_index('size', inplace=True)
    # hood_col_dict = {0:"총장", 1:"어깨너비", 2:"가슴단면", 3:"소매길이"}
    
    size_rec = find_mse_in_size_df(hood_size_df, user_prediction_lst)
    
    return render_template('Size_RecSys.html', pred='최종 추천 사이즈는 {}입니다!'.format(size_rec))
    

if __name__ == '__main__':
    app.run(debug=True)
