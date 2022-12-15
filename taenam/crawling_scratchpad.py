'''
Title : Crawling
Author : Taenam

변경사항 :
    - 리뷰 개수 100개 넘어야 돌아가는 코드 없앰
    - 상품 개수 +1 하는 과정 돌아가는 코드 안으로 넣음
    
Error : 
    - AttributeError: module 'collections' has no attribute 'Callable'
        https://stackoverflow.com/questions/69515086/error-attributeerror-collections-has-no-attribute-callable-using-beautifu

Reference :
    
'''

# %%
## import
import os
import time
import re
import math

import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from html_table_parser import parser_functions as parser

import collections
collections.Callable = collections.abc.Callable

import warnings
warnings.filterwarnings("ignore")

# %%
def get_review_content(driver, review_each_num):
    '''
    크롤링해서 내용 추출하는 함수 
    - 상품 클릭한 상태에서 해당 페이지 리뷰 10개 크롤링
    '''
    html = driver.page_source
    soup = BeautifulSoup(html,'lxml')
    
    # user, gender,height,weight,item, size,content, evaluation(size_eval,bright_eval,color_eval,thick_eval)
    user_list = []
    gender_list =[]
    height_list = []
    weight_list =[]
    item_list = []
    size_list = []
    star_list= []
    content_list = []
    size_eval_list =[]
    bright_eval_list =[]
    color_eval_list =[]
    thick_eval_list =[]
    
    for i in range(review_each_num):
        # profile(gender,height,weight)
        # <p class="review-profile__body_information">남성, 177cm, 85kg</p>
        try:
            profile_before = soup.find_all('p','review-profile__body_information')
            profile_after = profile_before[i].text.split(',')
            gender = profile_after[0]
            height = profile_after[1]
            weight = profile_after[2]
        except:
            gender = ''
            height = ''
            weight = ''
            
        # user :<p class="review-profile__name">LV 2 뉴비_95f88e16</p>           
        # item: #/<a href="https://www.musinsa.com/app/goods/1231416/0" class="review-goods-information__name">테이퍼드 히든 밴딩 크롭 슬랙스 [더스티 베이지]</a>
        # size :<span class="review-goods-information__option">
        # content
        try:
            user = soup.find_all('p','review-profile__name')[i].text
            item = soup.find_all('a','review-goods-information__name')[i].text
            size = soup.find_all('span', 'review-goods-information__option')[i].text.strip().replace('\n','') # '/n' 없애고 추출하기
            content = soup.find_all('div','review-contents__text')[i].text
        except:
            user = ''
            item = ''
            size = ''
            content = ''
            
        #star
        #->별 5개일때:<span class="review-list__rating__active" style="width: 100%"></span>
        #->별 4개일때:<span class="review-list__rating__active" style="width: 80%"></span>
        try:
            stars = driver.find_elements_by_xpath('//*[@id="reviewListFragment"]/div['+str(i+1)+']/div[3]/span/span/span')
            for j in stars:
                a =j.get_attribute('style')
                if a[7:9]=='20':
                    star = 1
                elif a[7:9]=='40':
                    star = 2
                elif a[7:9]=='60':
                    star = 3
                elif a[7:9]=='80':
                    star = 4
                else:
                    star = 5
        except:
            star = ''
      
        # evaluation
        try:
            evaluation = soup.find_all('div', 'review-evaluation')
            size_eval = evaluation[i].find_all('span')[0].text
            bright_eval = evaluation[i].find_all('span')[1].text
            color_eval = evaluation[i].find_all('span')[2].text
            thick_eval = evaluation[i].find_all('span')[3].text
        except:
            size_eval = ''
            bright_eval = ''
            color_eval = ''
            thick_eval = ''
        
        user_list.append(user)
        gender_list.append(gender)
        height_list.append(height)
        weight_list.append(weight)
        item_list.append(item)
        size_list.append(size)
        content_list.append(content)
        star_list.append(star)
        size_eval_list.append(size_eval)
        bright_eval_list.append(bright_eval)
        color_eval_list.append(color_eval)
        thick_eval_list.append(thick_eval)
    
    return user_list, gender_list, height_list, weight_list, item_list, size_list, content_list, star_list, size_eval_list, bright_eval_list, color_eval_list, thick_eval_list

def get_item_content(driver, reviewNum):
    lastPage = int(reviewNum/10)+1 # 리뷰 100개 이하일 때 마지막 페이지
    lastPage_review_num = int(reviewNum%10) # 리뷰 100개 이하일 때 마지막 페이지 리뷰 수
    
    # 첫 페이지
    page_cnt = 1
    print(page_cnt, end=" ")
    user_list, gender_list, height_list, weight_list, item_list, size_list, content_list, star_list, size_eval_list, bright_eval_list, color_eval_list, thick_eval_list = get_review_content(driver, 10)

    for i in range(2):
        for j in range(4, 9 ,1):
            
            if page_cnt >= 10: # 리뷰 100개만 긁어야하니 stop
                break
            
            if page_cnt == lastPage: # 리뷰가 100개 이하면 마지막 페이지에서 더 이상 못 넘기므로 stop
                break
            
            try:
                if j in [4,5,6,7]:
                    driver.find_element_by_css_selector('#reviewListFragment > div.nslist_bottom > div.pagination.textRight > div > a:nth-child(' + str(int(j)) + ')').send_keys(Keys.ENTER)
                elif j == 8:
                    driver.find_element_by_css_selector('#reviewListFragment > div.nslist_bottom > div.pagination.textRight > div > a.fa.fa-angle-right.paging-btn.btn.next').send_keys(Keys.ENTER)
                page_cnt += 1
                print(page_cnt, end=" ")
            except:
                print('{}, {} button click except'.format(i, j))
            time.sleep(2)

            
            # 첫 페이지 이외의 페이지 정보 첫 페이지 리스트에 추가
            # 마지막 페이지는 10개가 안될 수 있어 조건 부여
            if page_cnt == lastPage:
                user_list_1, gender_list_1, height_list_1, weight_list_1, item_list_1, size_list_1, content_list_1, star_list_1, size_eval_list_1, bright_eval_list_1, color_eval_list_1, thick_eval_list_1 = get_review_content(driver, lastPage_review_num)
            else:
                user_list_1, gender_list_1, height_list_1, weight_list_1, item_list_1, size_list_1, content_list_1, star_list_1, size_eval_list_1, bright_eval_list_1, color_eval_list_1, thick_eval_list_1 = get_review_content(driver, 10)
            
            user_list.extend(user_list_1)
            gender_list.extend(gender_list_1)
            height_list.extend(height_list_1)
            weight_list.extend(weight_list_1)
            item_list.extend(item_list_1)
            size_list.extend(size_list_1)
            content_list.extend(content_list_1)
            star_list.extend(star_list_1)
            size_eval_list.extend(size_eval_list_1)
            bright_eval_list.extend(bright_eval_list_1)
            color_eval_list.extend(color_eval_list_1)
            thick_eval_list.extend(thick_eval_list_1)
            
            time.sleep(2)

    item_review_df = pd.DataFrame({'user':user_list,
                                   'gender':gender_list,
                                   'height':height_list,
                                   'weight':weight_list,
                                   'item':item_list,
                                   'size':size_list,
                                   'star':star_list,
                                   'content':content_list,
                                   'size_eval':size_eval_list,
                                   'bright_eval':bright_eval_list,
                                   'color_eval':color_eval_list,
                                   'thick_eval':thick_eval_list})
            
    return item_review_df
    

def get_data(driver, start_item, item_cnt):
    final_df = pd.DataFrame() # item 90개 정보 담은 최종 DataFrame
    
    # 후기순 90개 자동으로 클릭하기

    for t in tqdm(range(start_item, item_cnt+1,1), desc = "Description"):
        print("Item {} : ".format(t), end = " ")
        ## 상품 클릭
        try: 
            # #searchList > li:nth-child(1) > div.li_inner > div.list_img > a > img
            # #searchList > li:nth-child(90) > div.li_inner > div.list_img > a > img
            driver.find_element_by_css_selector('#searchList > li:nth-child(' + str(int(t)) + ') > div.li_inner > div.list_img > a > img').click()
            time.sleep(2)
        except:
            print("item click issue")
            continue
        
        ## 팝업 처리
        # 일단 빈공간 클릭 -> 창이 랜덤으로 뜨기 때문에 빈곳을 클릭하면 팝업이 뜨는 경우가 있음
        driver.find_element_by_xpath('//*[@id="product_order_info"]/div[1]/h4')
        time.sleep(2)
        
        try:
            #무신사쿠폰 팝업창: 해당 팝업이 뜨면 나머지 선택 안됨
            driver.find_element_by_xpath('/html/body/div/div/div/button').click()
            time.sleep(2)
            #입고지연팝업창: 삭제안해도 나머지 구동 가능
            #driver.find_element_by_xpath('//*[@id="divpop_goods_niceghostclub_8451"]/form/button[2]').click()
            #무신사회원혜택 팝업창 :삭제안해도 나머지 구동가능
            #driver.find_element_by_xpath('//*[@id="page_product_detail"]/div[3]/div[23]/div/a[1]/img').click()
        except:
            pass
            
        ## 사이즈표 추출
        try:
            html = driver.page_source
            soup = BeautifulSoup(html,'html.parser')
            figure = soup.find('table',{'class':'table_th_grey'})
            p = parser.make2d(figure)
            figure_df = pd.DataFrame(data = p[1:],columns = p[0])
            figure_df.drop([0,1],inplace = True)
        except:
            print("사이즈표 오류발생")
            
        ## 리뷰개수 체크
        try:
            reviewNum = driver.find_element_by_xpath('//*[@id="estimate_style"]')
            reviewNum = reviewNum.text
            reviewNum = re.sub(r'[^0-9]','',reviewNum)
            reviewNum = int(reviewNum)
        except:
            print("리뷰개수 오류발생")
            reviewNum = 0
        
        ## 각 상품에 해당하는 리뷰 추출
        item_review_df = get_item_content(driver, reviewNum)
        
        merge_df = pd.merge(item_review_df, figure_df, how = 'left', left_on = 'size', right_on = 'cm')
        print(merge_df.shape)
        
        final_df = pd.concat([final_df, merge_df])
        
        #뒤로가기
        driver.back()
        time.sleep(2)

    return final_df

# %%
if __name__=='__main__':
    page = 4
    page_url = 'https://www.musinsa.com/categories/item/001004?d_cat_cd=001004&brand=&list_kind=small&sort=emt_high&sub_sort=&page=' + str(page) + '&display_cnt=90&group_sale=&exclusive_yn=&sale_goods=&timesale_yn=&ex_soldout=&kids=&color=&price1=&price2=&shoeSizeOption=&tags=&campaign_id=&includeKeywords=&measure='
    
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    options.add_experimental_option("detach", True) # 마지막에 창 안닫히게
    
    driver = webdriver.Chrome("./chromedriver.exe", options=options)
    
    driver.get(page_url)
    start_item = 14 # 1, 14
    item_cnt = 90
    # re_cnt = 100  # 기준으로 잡은 스타일후기리뷰 개수 (우선 생략)
    
    # 크롤링 진행
    final_df = get_data(driver, start_item, item_cnt)

    # 창 종료
    driver.quit()  
    
    final_df.drop_duplicates(subset = None, keep = 'first', inplace = True, ignore_index = True) #중복아이템 제거
    print('최종개수:',final_df.shape) #최종 data: final_df
    
    end = time.time()
    print("{:.5f} sec".format(end-start))

    # csv로 저장
    final_df.to_csv("../data/hood_4page.csv", encoding="UTF-8", index=False)
    
# %%
