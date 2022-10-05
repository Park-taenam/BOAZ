# %%
import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys

import warnings
warnings.filterwarnings("ignore")
# %%
def extraction_review_info(browser):
    name_list = [] # 구매 의류
    size_buy_list = [] # 구매 사이즈

    sex_list = [] # 성별
    height_list = [] # 키
    weight_list = [] # 몸무게

    size_review_list = [] # 사이즈 리뷰
    brightness_list = [] # 밝기 리뷰
    color_list = [] # 색감 리뷰
    thickness_list = [] # 두께감 리뷰

    review_list = [] # 리뷰 내용

    # 현재 페이지 정보
    html = browser.page_source
    soup = BeautifulSoup(html,'lxml')

    # 구매한 의류, 사이즈
    merchandise_info = soup.find_all('div', 'review-goods-information__item')
    for i in range(len(merchandise_info)):
        try:
            name_list.append(merchandise_info[i].text.replace(" ", "").strip().replace("\n", ",").split(',')[0])
            size_buy_list.append(merchandise_info[i].text.replace(" ", "").strip().replace("\n", ",").split(',')[3])
        except:
            name_list.append("NaN")
            size_buy_list.append("NaN")
        
    # 고객 성별, 키, 몸무게(없을 수도 있음)
    customer_info = soup.find_all('div', 'review-profile__information')
    for i in range(len(customer_info)):
        try:
            sex_list.append(customer_info[i].text.strip().split('\n')[0].split(',')[0])
        except:
            sex_list.append("NaN")
            
        try:
            height_list.append(customer_info[i].text.strip().split('\n')[0].split(',')[1])
        except:
            height_list.append("NaN")
        
        try:
            weight_list.append(customer_info[i].text.strip().split('\n')[0].split(',')[2])
        except:
            weight_list.append("NaN")

    # 리뷰(선택) - 사이즈, 밝기, 색감, 두께감
    review_choice = soup.find_all('ul', 'review-evaluation__list')
    for i in range(len(review_choice)):
        try:
            size_review_list.append(review_choice[i].text.strip().replace("\n", ",").replace(" ", ",").split(",")[1])
            brightness_list.append(review_choice[i].text.strip().replace("\n", ",").replace(" ", ",").split(",")[3])
            color_list.append(review_choice[i].text.strip().replace("\n", ",").replace(" ", ",").split(",")[5])
            thickness_list.append(review_choice[i].text.strip().replace("\n", ",").replace(" ", ",").split(",")[7])
        except:
            size_review_list.append("NaN")
            brightness_list.append("NaN")
            color_list.append("NaN")
            thickness_list.append("NaN")  
            
    # 리뷰
    reviews = soup.find_all('div', "review-contents__text")
    for i in range(len(reviews)):
        review_list.append(reviews[i].text)

    return name_list, size_buy_list, sex_list, height_list, weight_list, size_review_list, brightness_list, color_list, thickness_list, review_list

# %%
print("#"*20, "Start", "#"*20)
browser = webdriver.Chrome()
# browser.maximize_window()

# 지금은 5개만 했지만 나중에 순서대로 선택하는 코드도 추가!
url_list = ['https://www.musinsa.com/app/goods/1108007', 
            'https://www.musinsa.com/app/goods/1558197', 
            'https://www.musinsa.com/app/goods/1283757', 
            'https://www.musinsa.com/app/goods/965917',
            'https://www.musinsa.com/app/goods/1114716']

browser.get(url_list[0])
print("browser open")
time.sleep(5)

# 들어가자마자 뜨는 할인쿠폰 창 없애기
try:
    button = browser.find_element_by_class_name('ab-close-button')
    button.send_keys(Keys.ENTER)
    print("coupon close")
except:
    print("no coupon")
time.sleep(2)

# 첫 페이지
page_cnt = 1
print(page_cnt, end=" ")
name_list, size_buy_list, sex_list, height_list, weight_list, size_review_list, brightness_list, color_list, thickness_list, review_list = extraction_review_info(browser)
time.sleep(2)

# 페이지 넘기면서 추출
for i in range(12):
    for i in range(4, 9 ,1):
        try:
            path = '//*[@id="reviewListFragment"]/div[11]/div[2]/div/a[' + str(i) + ']'
            button = browser.find_element_by_xpath(path)
            button.send_keys(Keys.ENTER)
            page_cnt += 1
            print(page_cnt, end=" ")
        except:
            print('except')
        time.sleep(2)
        
        # 다른 페이지 정보 받아서 한 리스트에 합치기
        name_list_1, size_buy_list_1, sex_list_1, height_list_1, weight_list_1, size_review_list_1, brightness_list_1, color_list_1, thickness_list_1, review_list_1 = extraction_review_info(browser)
        
        name_list.extend(name_list_1)
        size_buy_list.extend(size_buy_list_1)
        sex_list.extend(sex_list_1)
        height_list.extend(height_list_1)
        weight_list.extend(weight_list_1)
        size_review_list.extend(size_review_list_1)
        brightness_list.extend(brightness_list_1)
        color_list.extend(color_list_1)
        thickness_list.extend(thickness_list_1)
        review_list.extend(review_list_1)
        time.sleep(2)

print("\n", "#"*20, "Done", "#"*20)

# Webdriver 종료
browser.quit()
# %%
review_df = pd.DataFrame({"name":name_list,
                          "size_buy":size_buy_list,
                          "sex":sex_list,
                          "height":height_list,
                          "weight":weight_list,
                          "size_review":size_review_list,
                          "brightness":brightness_list,
                          "color":color_list,
                          "thickness":thickness_list,
                          "review":review_list})
# %%
print(review_df.shape)
print(review_df.columns)
