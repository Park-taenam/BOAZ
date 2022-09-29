# %%
import pandas as pd
import numpy as np
import os, sys, gc
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')
# %%
# data = pd.read_pickle('data.pkl')
data_top1 = pd.read_csv('./data/mutandard_top1.csv', encoding='cp949', index_col=0)

def preprocessing(data):
    data['height'] = [int(height.strip().split('c')[0]) for height in data['height']]
    data['weight'] = [int(weight.strip().split('k')[0]) for weight in data['weight']]
    
    return data

data_top1 = preprocessing(data_top1)
# %%
## KNN Classifier
X = data_top1.loc[:, ['height', 'weight']].values
y = data_top1.loc[:, 'size'].values

scaler_x = StandardScaler()
scaler_x.fit(X)
X_scaled = scaler_x.transform(X)
plt.scatter(pd.DataFrame(X)[0], pd.DataFrame(X)[1])
plt.show()
plt.scatter(pd.DataFrame(X_scaled)[0], pd.DataFrame(X_scaled)[1])
plt.show()

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5) # 하이퍼파라미터 조정 필요
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %%

# %%
