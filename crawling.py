# %%
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
    html = driver.page_source
    soup = BeautifulSoup(html,'lxml')
    
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
    
        try:
            user = soup.find_all('p','review-profile__name')[i].text
            item = soup.find_all('a','review-goods-information__name')[i].text
            size = soup.find_all('span', 'review-goods-information__option')[i].text.strip().replace('\n','')
            content = soup.find_all('div','review-contents__text')[i].text
        except:
            user = ''
            item = ''
            size = ''
            content = ''
            
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
    lastPage = int(reviewNum/10)+1
    lastPage_review_num = int(reviewNum%10)
    
    page_cnt = 1
    print(page_cnt, end=" ")
    if page_cnt == lastPage:
        user_list, gender_list, height_list, weight_list, item_list, size_list, content_list, star_list, size_eval_list, bright_eval_list, color_eval_list, thick_eval_list = get_review_content(driver, lastPage_review_num)
    else:
        user_list, gender_list, height_list, weight_list, item_list, size_list, content_list, star_list, size_eval_list, bright_eval_list, color_eval_list, thick_eval_list = get_review_content(driver, 10)

    for i in range(2):
        for j in range(4, 9 ,1):
            if page_cnt >= 10:
                break
            if page_cnt == lastPage:
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

def get_data(driver, page_url, start_item, item_cnt):
    final_df = pd.DataFrame()

    for t in tqdm(range(start_item, item_cnt+1,1), desc = "Description"):
        print("Item {} : ".format(t), end = " ")
        try: 
            driver.find_element_by_css_selector('#searchList > li:nth-child(' + str(t) + ') > div.li_inner > div.list_img > a').click()
            time.sleep(2)
        except:
            print("item click issue")
            continue

        driver.find_element_by_xpath('//*[@id="product_order_info"]/div[1]/h4')
        time.sleep(2)
        
        try:
            driver.find_element_by_xpath('/html/body/div/div/div/button').click()
            time.sleep(2)
        except:
            pass
            
        try:
            html = driver.page_source
            soup = BeautifulSoup(html,'html.parser')
            figure = soup.find('table',{'class':'table_th_grey'})
            p = parser.make2d(figure)
            figure_df = pd.DataFrame(data = p[1:],columns = p[0])
            figure_df.drop([0,1],inplace = True)
        except:
            print("사이즈표 오류발생")
            
            
        try:
            reviewNum = driver.find_element_by_xpath('//*[@id="estimate_style"]')
            reviewNum = reviewNum.text
            reviewNum = re.sub(r'[^0-9]','',reviewNum)
            reviewNum = int(reviewNum)
        except:
            print("리뷰개수 오류발생")
            reviewNum = 0
        
        item_review_df = get_item_content(driver, reviewNum)
        
        merge_df = pd.merge(item_review_df, figure_df, how = 'left', left_on = 'size', right_on = 'cm')
        print(merge_df.shape)
        
        final_df = pd.concat([final_df, merge_df])
        
        driver.get(page_url)
        time.sleep(2)

    return final_df

# %%
if __name__=='__main__':
    page = 1 # 원하는 페이지 입력
    page_url = 'https://www.musinsa.com/categories/item/001004?d_cat_cd=001004&brand=&list_kind=small&sort=emt_high&sub_sort=&page={}&display_cnt=90&group_sale=&exclusive_yn=&sale_goods=&timesale_yn=&ex_soldout=&kids=&color=&price1=&price2=&shoeSizeOption=&tags=&campaign_id=&includeKeywords=&measure='.format(str(page))
    
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    options.add_experimental_option("detach", True) # 마지막에 창 안닫히게
    
    driver = webdriver.Chrome("./chromedriver.exe", options=options)
    
    driver.get(page_url)
    time.sleep(2)

    start = time.time()

    start_item = 1
    item_cnt = 90
    
    # 크롤링 진행
    final_df = get_data(driver, page_url, start_item, item_cnt)

    # 창 종료
    driver.quit()  

    print('최종개수:',final_df.shape)
    
    end = time.time()
    print("{:.5f} sec".format(end-start))

    # csv로 저장
    final_df.to_csv("../data/hood_{}page.csv".format(page), encoding="UTF-8", index=False)
