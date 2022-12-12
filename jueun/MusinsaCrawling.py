# %%
##################환경설정#################################
import warnings
warnings.filterwarnings("ignore")
from selenium import webdriver
from bs4 import BeautifulSoup
import time
import re
import math
import pandas as pd

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from html_table_parser import parser_functions as parser

# %%
 ###################함수정의###########################
 #user, gender,height,weight,item, size,content, evaluation(size_eval,bright_eval,color_eval,thick_eval)
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
#함수정의
def get_content(driver):
    #함수안에 html, soup 넣어놔야 페이지 넘어가서 바르게 긁어옴, 밖에 빼놓으면 첫페이지만 여러번 긁어진다.
    html = driver.page_source
    soup = BeautifulSoup(html,'lxml')
    for i in range(10):
    
    #profile(gender,height,weight)
    #<p class="review-profile__body_information">남성, 177cm, 85kg</p>
        profile_before = soup.find_all('p','review-profile__body_information')
        profile_after = profile_before[i].text.split(',')
        try:
            gender = profile_after[0]
            height = profile_after[1]
            weight = profile_after[2]
        except:
            gender = ''
            height = ''
            weight = ''
#user :<p class="review-profile__name">LV 2 뉴비_95f88e16</p>           
#item: #/<a href="https://www.musinsa.com/app/goods/1231416/0" class="review-goods-information__name">테이퍼드 히든 밴딩 크롭 슬랙스 [더스티 베이지]</a>
#size :<span class="review-goods-information__option">
#content

        try:
            user = soup.find_all('p','review-profile__name')[i].text
            item = soup.find_all('a','review-goods-information__name')[i].text
            # '/n' 없애고 추출하기
            size = soup.find_all('span', 'review-goods-information__option')[i].text.strip().replace('\n','')
            content = soup.find_all('div','review-contents__text')[i].text
        except:
            user = ''
            item = ''
            size = ''
            content = ''
            
        #star
#->별 5개일때:<span class="review-list__rating__active" style="width: 100%"></span>
#->별 4개일때:<span class="review-list__rating__active" style="width: 80%"></span>
        stars = driver.find_elements_by_xpath('//*[@id="reviewListFragment"]/div['+str(i+1)+']/div[3]/span/span/span')
        try:
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
      

    #evaluation
        evaluation = soup.find_all('div', 'review-evaluation')
        try:
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
  
   
        
#버튼 누르기 함수정의
def move_next(driver):    
    for p in range(4):
        get_content(driver)
        #페이지 2,3,4,5 넘어가기
        driver.find_element_by_css_selector('#reviewListFragment > div.nslist_bottom > div.pagination.textRight > div > a:nth-child(' + 
                                            str(int(4) + int(p)) + ')').send_keys(Keys.ENTER)
        time.sleep(2)
    get_content(driver)
    
#그다음 화살표'>'버튼누르기: (6,7,8...)있는 페이지로 넘어가기   
def move_arrow(driver):
    driver.find_element_by_css_selector('#reviewListFragment > div.nslist_bottom > div.pagination.textRight > div > a.fa.fa-angle-right.paging-btn.btn.next').send_keys(Keys.ENTER)


# %%
##########################크롤링시작##############################
#----------바꿀변수-------------#
 
final_df = pd.DataFrame() 
#8페이지
options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-logging"])
driver = webdriver.Chrome(options=options)
#--------------------------------본인해당 page url로바꾸기-----------------------#
driver.get('https://www.musinsa.com/categories/item/001004?d_cat_cd=001004&brand=&list_kind=small&sort=emt_high&sub_sort=&page=8&display_cnt=90&group_sale=&exclusive_yn=&sale_goods=&timesale_yn=&ex_soldout=&kids=&color=&price1=&price2=&shoeSizeOption=&tags=&campaign_id=&includeKeywords=&measure=')
time.sleep(3)

start = time.time()
math.factorial(100000)

#-----------------item_cnt: 상품개수 + 1로 바꾸기 #90개-> 91------------#
item_cnt = 91 
#-----------------range(시작아이템개수,,)바꾸기------------------------#
for t in range(1,item_cnt,1):
    #후기순 90개 자동으로 클릭하기
# #searchList > li:nth-child(1) > div.li_inner > div.list_img > a > img
# #searchList > li:nth-child(90) > div.li_inner > div.list_img > a > img
    try: 
        driver.find_element_by_css_selector('#searchList > li:nth-child(' +
                                            str(int(t)) +
                                            ') > div.li_inner > div.list_img > a > img').click()
        time.sleep(2)
    except:
        continue
#일단 빈공간 클릭->창이 랜덤으로 뜨기 때문에 빈곳을 클릭하면 팝업이 뜨는 경우가 있음
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
        
        #사이즈표
    try:
        html = driver.page_source
        soup = BeautifulSoup(html,'html.parser')
        figure = soup.find('table',{'class':'table_th_grey'})
        p = parser.make2d(figure)
        figure_df = pd.DataFrame(data = p[1:],columns = p[0])
        figure_df.drop([0,1],inplace = True)
    except:
        print("사이즈표 오류발생")
        
        #리뷰개수
    try:
        reviewNum = driver.find_element_by_xpath('//*[@id="estimate_style"]')
        reviewNum = reviewNum.text
        reviewNum = re.sub(r'[^0-9]','',reviewNum)
        reviewNum = int(reviewNum)
        
    except:
        print("리뷰개수 오류발생")
        reviewNum = 0
        

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
        
    #----------------- b:기준으로 잡은 스타일후기리뷰 개수,우리는 100개------------# 
    b = 100  
    
    if reviewNum >= b:
        #크롤링
        #1:50개, 2:100개, 10:500개, 
        for k in range(2):
            try:
                move_next(driver)
                move_arrow(driver)
        #move_next(driver)
            except:
                time.sleep(2)
    else:
        pass
        

    print(str(t),'번째제품 리뷰',reviewNum,'개')
    
    a = t      
    time.sleep(2)    
    globals()["df" + str(a)] = pd.DataFrame({'user':user_list,
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
    #사이즈표와 리뷰 merge
    globals()["merge_df"+str(a)] = pd.merge(globals()["df" + str(a)],figure_df,how = 'left',left_on = 'size',right_on = 'cm')
    print('df',a,'shape',globals()["merge_df"+str(a)].shape)
    
    final_df = pd.concat([final_df, globals()["merge_df"+str(a)]])
    #뒤로가기
    driver.back()
    time.sleep(2)
          


end = time.time()
print(f"{end - start:.5f} sec")
final_df.drop_duplicates(subset = None,keep = 'first', inplace = True,ignore_index = True) #중복아이템 제거
print('최종개수:',final_df.shape) #최종 data: final_df
driver.quit()  

# %%
#------------------파일이름바꾸기-----------------------------#
final_df.to_csv("data/78_90_8page.csv", encoding="UTF-8", index=False)
# %%
