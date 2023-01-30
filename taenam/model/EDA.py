# %%
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
# %%
data_hood = pd.read_csv('../data/hood_page1to20_final_nodup.csv')
data_hood = data_hood.iloc[:, 1:]

data_hood_null = data_hood[data_hood['총장'].isnull() & data_hood['어깨너비'].isnull() & data_hood['가슴단면'].isnull() & data_hood['소매길이'].isnull()].reset_index()
# %%
## EDA
print("중복 제거한 데이터  : {}\n".format(data_hood.shape[0]))

print("총장 is null : {}".format(data_hood[data_hood['총장'].isnull()].shape[0]))
print("어깨너비 is null : {}".format(data_hood[data_hood['어깨너비'].isnull()].shape[0]))
print("가슴단면 is null : {}".format(data_hood[data_hood['가슴단면'].isnull()].shape[0]))
print("소매길이 is null : {}".format(data_hood[data_hood['소매길이'].isnull()].shape[0]))
print("수치 모두 is null : {}".format(data_hood[data_hood['총장'].isnull() & data_hood['어깨너비'].isnull() & data_hood['가슴단면'].isnull() & data_hood['소매길이'].isnull()].shape[0]))
print("수치 적어도 하나 is null : {}\n".format(data_hood[data_hood['총장'].isnull() | data_hood['어깨너비'].isnull() | data_hood['가슴단면'].isnull() | data_hood['소매길이'].isnull()].shape[0]))

print("성별 is null : {}".format(data_hood[data_hood['gender'].isnull()].shape[0]))
print("키 is null : {}".format(data_hood[data_hood['height'].isnull()].shape[0]))
print("몸무게 is null : {}".format(data_hood[data_hood['weight'].isnull()].shape[0]))
print("user 정보 모두 is null : {}".format(data_hood[data_hood['gender'].isnull() & data_hood['height'].isnull() & data_hood['weight'].isnull()].shape[0]))
print("user 정보 적어도 하나 is null : {}".format(data_hood[data_hood['gender'].isnull() | data_hood['height'].isnull() | data_hood['weight'].isnull()].shape[0]))

# %%
## About crawling problem
print("수치 없는 사이즈 갯수 : {}".format(len(data_hood_null['size'].unique())))

spe_size_lst_1 = [x for x in data_hood_null['size'].unique() if "," not in x] # , 있으면 제거 (셋업 or 세트)
print(", 없는 사이즈 갯수 : {}".format(len(spe_size_lst_1)))

spe_size_lst_2 = [x for x in spe_size_lst_1 if "/" in x]
print(", 없고 / 있는 사이즈 갯수 : {}".format(len(spe_size_lst_2)))

spe_size_lst_3 = [x for x in spe_size_lst_1 if "/" not in x]
print(", 없고 / 없는 사이즈 갯수 : {}".format(len(spe_size_lst_3)))

print()

df = pd.DataFrame(data_hood_null['size'].value_counts())
print("수치 없는 사이즈 상품 수 : {}".format(sum(df['size'])))

df_1 = df.loc[spe_size_lst_1, :].sort_values(by='size')
print(", 없는 사이즈 상품 수 : {}".format(sum(df_1['size'])))

df_2 = df.loc[spe_size_lst_2, :].sort_values(by='size') # 얘가 중요
print(", 없고 / 있는 사이즈 상품 수 : {}".format(sum(df_2['size'])))

df_3 = df.loc[spe_size_lst_3, :].sort_values(by='size')
print(", 없고 / 없는 사이즈 상품 수 : {}".format(sum(df_3['size'])))

## check
size_test_1 = [x.split('/')[0].strip() for x in spe_size_lst_2]
print(size_test_1.count("S"))
print(size_test_1.count("M"))
print(size_test_1.count("L"))
print(size_test_1.count("XL"))

size_test_2 = [x.split('/')[1].strip() for x in spe_size_lst_2]
print(size_test_2.count("S"))
print(size_test_2.count("M"))
print(size_test_2.count("L"))
print(size_test_2.count("XL"))

# %%
## preprocessing
data_hood = data_hood[data_hood['gender'].notnull() & data_hood['height'].notnull() & data_hood['weight'].notnull()]
data_hood = data_hood[data_hood['총장'].notnull() | data_hood['어깨너비'].notnull() | data_hood['가슴단면'].notnull() | data_hood['소매길이'].notnull()]
data_hood.reset_index(inplace=True)