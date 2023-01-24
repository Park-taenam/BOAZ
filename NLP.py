# %%
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
# %%
def WordDictionary():
    total = ['사이즈', '사이즈하', '사이즈감', '임사이즈', '여사이즈', '사이즌', '뻐요사이즈', '싸이즈', '핏이', '핏', '핏입니다',
             '핏감도', '핏나', '핏감이랑', '핏처럼', '핏입니당', '핏이에요', '핏이예요', '핏이랑', '핏이구요', '핏이네요', '핏감은',
             '핏입니', '핏이라서', '핏으로', '옷핏', '핏더', '핏도', '핏감', '옷', '후드', '후드티', '핏은', '좋아요핏', '요핏도',
             '좋아요핏도', '상의', '요핏', '폼', '폭', '사이즈', '후드', '옷', '품', '전체', '품도', '몸통', '옷핏', '상체', '핏',
             '핏이구', '핏이었', '핏임', '핏입','핏입니', '핏감', '요핏', '폼', '후드핏', '핏이', '크기', '상품', '상의', '핏더']
    
    chongjang = ['길이', '기장', '총장', '궁디', '궁딩', '엉덩이', '밑', '밑단', '위아래', '끝단', '밑기장',
                 '밑단이랑', '기장이', '기장입니', '밑단', '밑', '기장감', '요기장', '총길이', '총장', '총기장',
                 '길이감', '끝', '기장', '길이']
    
    shoulder = ['어깨핏', '어께', '어깨', '너비', '어깡', '골격', '바디', '가로', '넓이', '등판', '통',
                '어깨깡패', '어깨라', '어깨넓', '어깨쪽', '어께', '몸집', '어깨핏', '어깨선', '어깨라인', '어깡',
                '어깨', '통', '몸통', '골격', '넓이', '너비', '등판']

    chest = ['가슴', '둘레', '바디', '가로', '넓이', '통', '둘레', '몸집', '가슴팍', '통', '몸통']
    
    arm = ['소매', '팔도', '손목', '손', '팔다리', '팔목', '팔', '팔꿈치', '팔이', '요팔',
           '당팔', '팔길이', '팔다리', '팔길', '팔소매', '손목', '팔목', '소매통', '손', '팔', '소매', '팔기장']

    small = ['작습니다', '짧았어요', '짧습니다', '짧으면', '짧', '작았어요', '짧다고', '쪼여서', '작았으면', '작아도', '작으면', '작음', '작아진', '좁다고',
             '짧아요', '짧게', '짧다는', '좁다는', '작네', '작기는', '달라붙는', '작은데', '짧은데', '짧음',
             '작긴', '짧네요', '작게', '작아서', '작은', '짧고', '짧아', '좁은', '타이트', '짧았지만', '붙어서', '짧긴',
             '작은것', '작아요', '짧은', '짧아서', '짧지만', '좁아', '좁고', '작네요', '작아', '쪼이는', '좁아요', '크롭느낌',
             '크롭하', '작고', '숏', '쫄려', '작다고', '작다', '좁아서', '작', '크롭', '크롭된', '미니', '크롭해',
             '길었으면', '컸으면', '좁게', '짧지', '작지', '크롭이', '핏되',
             '숏하', '사이즈업하', '크롭하', '숏한', '크롭하긴', '크롭이', '크롭해', '크롭느낌', '크롭된', '사이즈업', '짝',
             '작', '숏', '짧긴', '크롭', '크롭한', '타이트', '크롭이라', '작긴', '작', '짧', '좁']

    big = ['큰', '긴', '커요', '크고', '크긴', '넉넉하고', '길어서', '넉넉하게', '작다는'
           '넉넉한', '클', '길고', '큰데', '크지', '크지만', '부해', '크니깐', '크더라구요', '컸어도', '넓음', '큼직하게',
           '박시하네요', '넉넉할', '헐렁하게', '큼직하니', '컸음', '큽니', '긴데', '길어', '길게', '넓어서', '넓어', '박시하고', '컸어요',
           '박시', '크다고', '크다는', '넓게', '넓고', '커용', '길다는', '크지도', '큼', '크다', '길었어요', '넉넉함', '큰데도', '덮는데',
           '덮이고', '컸고', '벙벙해서', '크더라고요', '컸을', '크거나', '크더라', '컸었는데', '컸어용', '컸습니다', '덮네요',
           '큼지막해서', '덥는', '박시하게', '크네요', '클까', '넉넉하구', '박시하', '컸는데', '넉넉하네요', '크게', '크니까', '넉넉한데', '길어요',
           '컸지만', '헐렁한', '크네', '넉넉해', '펑퍼짐한', '덮어용', '박시하니', '헐렁해', '넉넉하지''넉넉하구요', '커', '큽니다', '넉넉하니',
           '크', '길지', '박시한데', '덮어요', '길긴', '넉넉해요', '넓은', '덮히는', '넓긴', '덮어줘서', '덮은', '넉넉해서', '큼직해서', '기네', '낙낙',
           '흘러내리는', '덮어서', '벙벙한', '커서', '큰것', '박시핏', '오퍼핏', '낙낙해', '여유핏', '널널함', '길', '덮어', '넉넉합니다', '크면',
           '덮고', '가오리', '루즈핏', '펑퍼짐', '길었지만', '헐렁하고', '남아', '아방핏', '박스핏', '널널', '오바핏', '덮습니다', '넉넉히',
           '헐렁해서', '접어도', '접어', '접어서', '널', '넓어요', '덮고도', '오버', '와이드', '낭낭', '헐렁', '접으면', '짧았으면', '접고',
           '버핏', '접어야', '잡아먹힌', '수선해서', '오버핏', '오버사이즈', '오버사이징해서', '낙낙하다', '낙낙한데', '낙낙해요', '오버핏이예요', '오버핏입니당',
           '요박시하', '컸어용', '낙낙합니', '어벙벙하', '와이드핏', '크니깐', '크네요옷', '널널해요', '오버핏나와요', '널럴하',
           '요넉넉하', '엄청큼', '여유핏', '오버느낌', '아방핏', '오버핏은', '넓긴', '넉넉함', '입니다오버핏', '큽니다오버핏',
           '큽니다제', '큽니당', '오버해요', '요박시한', '헐렁하긴', '헐렁거리', '아방방한', '아방방', '박시합니', '박시했다',
           '빅사이즈', '받았습니다오버핏', '빡시하', '루즈핏으로', '낭낭한', '오버핏이긴', '박스핏', '널널함', '어벙벙',
           '요넉넉한', '오퍼핏', '오버핏이구요', '낙낙해', '낭낭해서', '넉넉하구', '넉넉한데', '아방해', '오버핏이라서',
           '낭낭하', '박시해', '커용', '넉넉해', '펑퍼짐', '박시하지', '널널', '좋아요오버핏', '어벙벙한', '널널하', '오바핏',
           '오버하지', '오버핏하', '박시합니다', '오버핏되', '박시함', '버핏', '박시한', '낙낙하니', '박시해요', '크네용',
           '낙낙해서', '큼직해서', '루즈해', '느슨', '널널합니다', '아방아', '아방아방', '루즈핏이', '박시하', '세미오버',
           '세미오버핏', '루즈해서', '오버핏이라', '루즈핏이라', '오버핏이네요', '오버핏입니', '어벙', '오버해', '박시한데',
           '아방', '오버핏이에요', '예뻐요오버핏', '큼지막', '박시핏', '오버해서', '루즈한', '오버핏인데', '넉넉하구요', '박시할',
           '큼직', '오버한', '오버핏이', '낙낙하', '아방하', '롱', '오버핏으로', '낙낙한', '오버핏입니다', '박시해서', '와이드',
           '요오버핏', '넉넉해서', '큽니', '오버', '오버하', '루즈', '루즈핏', '박시', '넉넉', '길', '크긴', '아방한', '널널한',
           '헐렁', '길긴', '와이드하', '아방해서', '세미오버사이즈', '크', '길', '널', '덮', '넓', '벙벙하', '덮이', '덮히', '커다랗', '덥히', '넉넉해지', '뒤집', '넉넉']
    
    return total, chongjang, shoulder, chest, arm, small, big
    
def WordDict_nodup(word_lst):
    # 언어사전 내 겹치는 단어 제거
    print('언어사전 중복제거 전:{}개'.format(len(word_lst)), end=" ")
    word_lst = set(word_lst)
    print('중복제거 후:{}개'.format(len(word_lst)))

    return list(word_lst)

def get_keywords(keywords, df):
    review = df['review']
    df['keyword_in_review'] = str('')
    
    for i in tqdm(range(len(review))): 
        keywords_search = []
        for j in keywords:
            if re.findall(j, review[i]):
                a = re.findall(j +'+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+',review[i]) # 키워드 + 단어
                aa = re.findall(j + ' '+'+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+',review[i]) # 키워드 + 띄고 단어
                b = re.findall('[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+' + j ,review[i]) # 단어 + 키워드
                bb = re.findall('[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+' + j ,review[i]) # 단어 + 띄고 키워드
                
                # 앞에 지만/데/고/보다 등이 들어있으면 키워드에 대한 수식어구가 아니므로 처리
                if(len(b)!=0):
                    if(('지만' in b[0])|('데' in b[0])|('고' in b[0])|('보다' in b[0])|
                       ('도' in b[0])|('서' in b[0])|('요' in b[0])):
                        b=[]
                    else:
                        pass
                else:
                    pass
                if(len(bb)!=0):
                    if(('지만' in bb[0])|('데' in bb[0])|('고' in bb[0])|('보다' in bb[0])|
                       ('도' in bb[0])|('서' in bb[0])|('요' in bb[0])):
                        bb=[]
                    else:
                        pass
                    
                keywords_search.extend(a)
                keywords_search.extend(aa)
                keywords_search.extend(b)
                keywords_search.extend(bb)

        if len(keywords_search) != 0:
            keywords_o = ','.join(x for x in keywords_search)
            df.loc[i, 'keyword_in_review'] = keywords_o 
        else:
            df.loc[i, 'keyword_in_review'] = None
    
    return df['keyword_in_review']

def big_small(big, small, df, name_big_small):
    review = df[name_big_small+'_keyword']
    df[name_big_small + '_big'] = str('')
    df[name_big_small + '_small'] = str('')

    for i in tqdm(range(len(review))):
        big_search = []
        small_search = []
        for j in big:
            if re.findall(j, str(review[i])):
                a = re.findall(j, str(review[i]))  # 키워드
                big_search.extend(a)

        for j in small:
            if re.findall(j, str(review[i])):
                a = re.findall(j, str(review[i]))  # 키워드
                small_search.extend(a)

        if len(big_search) != 0:
            df.loc[i, name_big_small + '_big'] = 1
        else:
            df.loc[i, name_big_small + '_big'] = 0

        if len(small_search) != 0:
            df.loc[i, name_big_small + '_small'] = 1
        else:
            df.loc[i, name_big_small + '_small'] = 0

    return df

def remove_both_one(df):
    total_both = (df['total_big'] == 1) & (df['total_small'] == 1)
    chongjang_both = (df['chongjang_big'] == 1) & (df['chongjang_small'] == 1)
    shoulder_both = (df['shoulder_big'] == 1) & (df['shoulder_small'] == 1)
    chest_both = (df['chest_big'] == 1) & (df['chest_small'] == 1)
    arm_both = (df['arm_big'] == 1) & (df['arm_small'] == 1)
    both = (total_both | chongjang_both |
            shoulder_both | chest_both | arm_both)

    print('전체: {}, 총장: {}, 가슴: {}, 소매: {}, 어깨: {}, 11 관측치: {}'.format(total_both.sum(
    ), chongjang_both.sum(), chest_both.sum(), arm_both.sum(), shoulder_both.sum(), both.sum()))

    df.drop(df[both].index, inplace=True)
    df.reset_index(inplace=True)

    return df

def merge_total_others(df):
    for i in range(df.shape[0]):
        if(df['total_big'][i] == 1):
            df['chongjang_big'][i] = 1
            df['shoulder_big'][i] = 1
            df['chest_big'][i] = 1
            df['arm_big'][i] = 1
        else:
            pass

        if(df['total_small'][i] == 1):
            df['chongjang_small'][i] = 1
            df['shoulder_small'][i] = 1
            df['chest_small'][i] = 1
            df['arm_small'][i] = 1
            
    #둘다 1인거 몇개인지확인
    print('전체 : {}'.format(df[(df['total_big']==1)&(df['total_small']==1)].shape[0]))
    print('어깨 : {}'.format(df[(df['shoulder_big']==1)&(df['shoulder_small']==1)].shape[0]))
    print('가슴 : {}'.format(df[(df['chest_big']==1)&(df['chest_small']==1)].shape[0]))
    print('소매 : {}'.format(df[(df['arm_big']==1)&(df['arm_small']==1)].shape[0]))
    print('총장 : {}'.format(df[(df['chongjang_big']==1)&(df['chongjang_small']==1)].shape[0]))

    shoulder  = (df['shoulder_big']==1)&(df['shoulder_small']==1)
    chest     = (df['chest_big']==1)&(df['chest_small']==1)
    arm       = (df['arm_big']==1)&(df['arm_small']==1)
    chongjang = (df['chongjang_big']==1)&(df['chongjang_small']==1)

    shoulder_idx = df.loc[shoulder, :].index
    chest_idx    = df.loc[chest,    :].index
    arm_idx      = df.loc[arm,      :].index
    chongjang_idx= df.loc[chongjang,:].index
    
    # 11 데이터 처리
    #총장
    df.loc[chongjang_idx[65],['chongjang_big','chongjang_small']] = 0,0 #19633
    df.loc[chongjang_idx[64],'chongjang_small'] = 0 #19344
    df.loc[chongjang_idx[63],['chongjang_big','chongjang_small']] = 0,0 #18989
    df.loc[chongjang_idx[62],'chongjang_small'] = 0 #18619
    df.loc[chongjang_idx[61],['chongjang_big','chongjang_small']] = 0,0 #18122
    df.loc[chongjang_idx[60],['chongjang_big','chongjang_small']] = 0,0 #17490
    df.loc[chongjang_idx[59],'chongjang_big'] = 0 #17058
    df.loc[chongjang_idx[58],'chongjang_small'] = 0 #16693
    df.loc[chongjang_idx[57],'chongjang_big'] = 0 #16063
    df.loc[chongjang_idx[56],'chongjang_big'] = 0 #15776
    df.loc[chongjang_idx[55],'chongjang_big'] = 0 #15761
    df.loc[chongjang_idx[54],'chongjang_big'] = 0 #15583
    df.loc[chongjang_idx[53],'chongjang_big'] = 0 #15282
    df.loc[chongjang_idx[52],'chongjang_big'] = 0 #14787
    df.loc[chongjang_idx[51],'chongjang_big'] = 0 #14326
    df.loc[chongjang_idx[50],'chongjang_small'] = 0 #14259
    df.loc[chongjang_idx[49],['chongjang_big','chongjang_small']] = 0,0 #14055
    df.loc[chongjang_idx[48],['chongjang_big','chongjang_small']] = 0,0 #14051
    df.loc[chongjang_idx[47],'chongjang_big'] = 0 #13987
    df.loc[chongjang_idx[46],'chongjang_small'] = 0 #13844
    df.loc[chongjang_idx[45],['chongjang_big','chongjang_small']] = 0,0 #13819
    df.loc[chongjang_idx[44],['chongjang_big','chongjang_small']] = 0,0 #13096
    df.loc[chongjang_idx[43],'chongjang_big'] = 0 #13089
    df.loc[chongjang_idx[42],'chongjang_big'] = 0 #13055
    df.loc[chongjang_idx[41],'chongjang_big'] = 0 #12782
    df.loc[chongjang_idx[40],'chongjang_big'] = 0 #12457
    df.loc[chongjang_idx[39],'chongjang_big'] = 0 #11711
    df.loc[chongjang_idx[38],'chongjang_big'] = 0 #11694
    df.loc[chongjang_idx[37],'chongjang_big'] = 0 #11682
    df.loc[chongjang_idx[36],'chongjang_big'] = 0 #10991
    df.loc[chongjang_idx[35],'chongjang_big'] = 0 #100801
    df.loc[chongjang_idx[34],'chongjang_big'] = 0 #10787
    df.loc[chongjang_idx[33],'chongjang_big'] = 0 #10650
    df.loc[chongjang_idx[32],'chongjang_small'] = 0 #10591
    df.loc[chongjang_idx[31],'chongjang_big'] = 0 #10533
    df.loc[chongjang_idx[30],'chongjang_big'] = 0 #10516
    df.loc[chongjang_idx[29],['chongjang_big','chongjang_small']] = 0,0 #10072
    df.loc[chongjang_idx[28],'chongjang_big'] = 0 #8761
    df.loc[chongjang_idx[27],'chongjang_big'] = 0 #8687
    df.loc[chongjang_idx[26],'chongjang_big'] = 0 #8508
    df.loc[chongjang_idx[25],'chongjang_small'] = 0 #8379
    df.loc[chongjang_idx[24],'chongjang_small'] = 0 #8268
    df.loc[chongjang_idx[23],'chongjang_big'] = 0 #7962
    df.loc[chongjang_idx[22],'chongjang_big'] = 0 #7479
    df.loc[chongjang_idx[21],'chongjang_big'] = 0 #7457
    df.loc[chongjang_idx[20],'chongjang_big'] = 0 #7327
    df.loc[chongjang_idx[19],'chongjang_big'] = 0 #7309
    df.loc[chongjang_idx[18],'chongjang_big'] = 0 #6906
    df.loc[chongjang_idx[17],['chongjang_big','chongjang_small']] = 0,0 #6649
    df.loc[chongjang_idx[16],'chongjang_small'] = 0 #6452
    df.loc[chongjang_idx[15],'chongjang_big'] = 0 #6404
    df.loc[chongjang_idx[14],'chongjang_big'] = 0 #6251
    df.loc[chongjang_idx[13],'chongjang_big'] = 0 #6201
    df.loc[chongjang_idx[12],'chongjang_big'] = 0 #6200
    df.loc[chongjang_idx[11],'chongjang_big'] = 0 #5654
    df.loc[chongjang_idx[10],'chongjang_big'] = 0 #5640
    df.loc[chongjang_idx[9],'chongjang_big'] = 0 #4885
    df.loc[chongjang_idx[8],'chongjang_small'] = 0 #4458
    df.loc[chongjang_idx[7],'chongjang_big'] = 0 #4162
    df.loc[chongjang_idx[6],'chongjang_big'] = 0 #4100
    df.loc[chongjang_idx[5],'chongjang_small'] = 0 #4003
    df.loc[chongjang_idx[4],['chongjang_big','chongjang_small']] = 0,0 #3776
    df.loc[chongjang_idx[3],'chongjang_big'] = 0 #3341
    df.loc[chongjang_idx[2],'chongjang_big'] = 0 #2182
    df.loc[chongjang_idx[1],'chongjang_big'] = 0 #1022
    df.loc[chongjang_idx[0],'chongjang_big'] = 0 #1019

    #소매
    df.loc[arm_idx[0],['arm_big','arm_small']] = 0 #3776
    df.loc[arm_idx[1],'arm_small'] = 0 #3973
    df.loc[arm_idx[2],'arm_small'] = 0 #8379
    df.loc[arm_idx[3],'arm_small'] = 0 #8647
    df.loc[arm_idx[4],'arm_small'] = 0 #14334
    df.loc[arm_idx[5],'arm_small'] = 0 #14787
    df.loc[arm_idx[6],'arm_small'] = 0 #14822
    df.loc[arm_idx[7],'arm_small'] = 0 #17135
    df.loc[arm_idx[8],'arm_small'] = 0 #17267
    df.loc[arm_idx[9],'arm_small'] = 0 #17490
    df.loc[arm_idx[10],'arm_small'] = 0 #17582
    df.loc[arm_idx[11],'arm_small'] = 0 #19344

    #가슴
    df.loc[chest_idx[0],'chest_small'] = 0 #2291
    df.loc[chest_idx[1],'chest_small'] = 0 #3481
    df.loc[chest_idx[2],['chest_small','chest_big']] = 0,0 #3891
    df.loc[chest_idx[3],'chest_small'] = 0 #3973
    df.loc[chest_idx[4],['chest_small','chest_big']] = 0,0 #5856
    df.loc[chest_idx[5],['chest_small','chest_big']] = 0,0 #5858
    df.loc[chest_idx[6],'chest_small'] = 0 #6651
    df.loc[chest_idx[7],'chest_small'] = 0 #8647
    df.loc[chest_idx[8],['chest_small','chest_big']] = 0 #14822
    df.loc[chest_idx[9],'chest_big'] = 0 #15483
    df.loc[chest_idx[10],['chest_big','chest_small']] = 0,0 #16142

    #어깨
    df.loc[shoulder_idx[0],'shoulder_small'] = 0 #2291
    df.loc[shoulder_idx[1],'shoulder_small'] = 0 #3481
    df.loc[shoulder_idx[2],'shoulder_small'] = 0 #3891
    df.loc[shoulder_idx[3],'shoulder_big']   = 0 #3905
    df.loc[shoulder_idx[4],'shoulder_small'] = 0 #3973
    df.loc[shoulder_idx[5],['shoulder_big','shoulder_small']] = 0,0 #5856
    df.loc[shoulder_idx[6],['shoulder_big','shoulder_small']] = 0,0 #5858
    df.loc[shoulder_idx[7],'shoulder_small'] = 0 #6651
    df.loc[shoulder_idx[8],'shoulder_small'] = 0 #8647
    df.loc[shoulder_idx[9],'shoulder_big'] = 0 #11917
    df.loc[shoulder_idx[10],['shoulder_big','shoulder_small']] = 0,0 #14822
    df.loc[shoulder_idx[11],'shoulder_big'] = 0 #15483
    df.loc[shoulder_idx[12],['shoulder_big','shoulder_small']] = 0,0 #16054
    df.loc[shoulder_idx[13],['shoulder_big','shoulder_small']] = 0,0 #16142
    df.loc[shoulder_idx[14],['shoulder_big','shoulder_small']] = 0,0 #17721
    df.loc[shoulder_idx[15],'shoulder_small'] = 0 #18183

    return df

# %%
if __name__=="__main__":
    
    data = pd.read_pickle('data_preprocessing_done.pkl')

    # 언어사전
    total, chongjang, shoulder, chest, arm, small, big = WordDictionary()

    total = WordDict_nodup(total)
    chongjang = WordDict_nodup(chongjang)
    shoulder = WordDict_nodup(shoulder)
    chest = WordDict_nodup(chest)
    arm = WordDict_nodup(arm)
    small = WordDict_nodup(small)
    big = WordDict_nodup(big)

    # 키워드 추출
    total_lst = get_keywords(total, data)
    chongjang_lst = get_keywords(chongjang, data)
    shoulder_lst = get_keywords(shoulder, data)
    chest_lst = get_keywords(chest, data)
    arm_lst = get_keywords(arm, data)

    data = pd.concat([data, total_lst, chongjang_lst,
                     shoulder_lst, chest_lst, arm_lst], axis=1)
    data.columns = ['user', 'gender', 'height', 'weight', 'item', 'size', 'star', 'content',
                    'size_eval', 'bright_eval', 'color_eval', 'thick_eval', 'cm', '총장',
                    '어깨너비', '가슴단면', '소매길이', 'review', 'keyword_in_review',
                    'total_keyword', 'chongjang_keyword', 'shoulder_keyword', 'chest_keyword', 'arm_keyword']
    data = data.drop(['keyword_in_review'], axis=1)

    # big, small 표현 탐지
    data = big_small(big, small, data, 'total')
    data = big_small(big, small, data, 'chongjang')
    data = big_small(big, small, data, 'arm')
    data = big_small(big, small, data, 'chest')
    data = big_small(big, small, data, 'shoulder')
    
    # big, small 둘 다 1인 행 삭제
    data = remove_both_one(data)
    
    # total과 총장,어깨,가슴,소매 통합
    data = merge_total_others(data)

    data.to_pickle('data_review_preprocessing_done.pkl')