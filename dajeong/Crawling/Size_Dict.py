# %%
###################################
#필요 라이브러리 로드
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
import pandas as pd
from tqdm.notebook import tqdm
import time, urllib
import numpy as np
import re, os, sys, json
from pykospacing import Spacing #띄어쓰기
spacing=Spacing()
from konlpy.tag import Kkma #형태소분석
from konlpy.tag import Okt #형태소분석
import soynlp; from soynlp.normalizer import * #정규화
from hanspell import spell_checker #맞춤법
from konlpy.tag import Komoran, Hannanum, Kkma, Okt
komoran = Komoran(); hannanum = Hannanum(); kkma = Kkma(); okt = Okt()

# %%
###################################
#데이터 불러오기
dfno = pd.read_pickle("data/crawlingdata_preprocess_done.pkl")
dfno.reset_index(drop=True, inplace=True)
dfno['review'] = str('')  #전처리된 리뷰 컬럼 정의


# %%
###################################
#리뷰 전처리
#(1)외국어 리뷰 탐색 및 삭제
i = 0
nohangul = []
for i in range(dfno.shape[0]):
    text = re.sub('[^ㄱ-ㅣ가-힣]', '', dfno.iloc[i,8])
    if(text==''):
        nohangul.append(i)
    else:
        pass
dfno = dfno.iloc[[True if i not in nohangul else False for i in range(dfno.shape[0])],:]
dfno.reset_index(drop=True, inplace=True)

#(2)자연어 전처리
i=0
for i in range(dfno.shape[0]):
    text = dfno.iloc[i,8]
    text = re.sub(pattern='[^\w\s\n]', repl='', string=text) #특수문자 제거
    text = re.sub(pattern='[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z]', repl=' ', string=text) #숫자, 그이외 삭제
    text = re.sub(pattern='[ㄱ-ㅎㅏ-ㅣ]+', repl='', string=text) #단순 모음, 자음 삭제
    text = repeat_normalize(text, num_repeats=2) #불필요반복문자정규화
    text = spacing(text) #띄어쓰기
    dfno['review'][i] = text


# %%
###################################
#형태소 분석
#(1)전처리된 리뷰만 불러오기
df = pd.DataFrame({'only_spacing':df['review']})

#(2)분석기별 사용할 품사 정리
#okt
chaeon_okt_pos = ['Noun','Unknown']
yongon_okt_pos = ['Adjective','Verb']
gwan_okt_pos = ['Determiner']
busa_okt_pos = ['Adverb']
chaeon_okt, yongon_okt, gwan_okt, busa_okt = [],[],[],[]

#hannanum
chaeon_hannanum_pos = ['N','NB','NC','NN','NP','NQ']
yongon_hannanum_pos = ['P','PA','PV','PX']
gwan_hannanum_pos = ['M','MM']
busa_hannanum_pos = ['M','MA']
chaeon_hannanum, yongon_hannanum, gwan_hannanum, busa_hannanum = [],[],[],[]

#(3)분석기별 토큰화 진행
#
for i in df['only_spacing']:
    text_hannanum = hannanum.pos(i)
    for one in range(len(text_hannanum)):
        if(text_hannanum[one][1]) in chaeon_hannanum_pos:
            chaeon_hannanum.append(text_hannanum[one][0])
        elif(text_hannanum[one][1]) in yongon_hannanum_pos:
            yongon_hannanum.append(text_hannanum[one][0])
        elif(text_hannanum[one][1]) in gwan_hannanum_pos:
            gwan_hannanum.append(text_hannanum[one][0])
        elif(text_hannanum[one][1]) in busa_hannanum_pos:
            busa_hannanum.append(text_hannanum[one][0])
        else:
            pass
for i in df['only_spacing']:
    text_okt = okt.pos(i)
    for one in range(len(text_okt)):
        if(text_okt[one][1]) in chaeon_okt_pos:
            chaeon_okt.append(text_okt[one][0])
        elif(text_okt[one][1]) in yongon_okt_pos:
            yongon_okt.append(text_okt[one][0])
        elif(text_okt[one][1]) in gwan_okt_pos:
            gwan_okt.append(text_okt[one][0])
        elif(text_okt[one][1]) in busa_okt_pos:
            busa_okt.append(text_okt[one][0])
        else:
            pass

#(4)(Okt+Hannanum)언어사전 정의
total = ['사이즈','사이즈하','사이즈감','임사이즈','여사이즈','사이즌','뻐요사이즈','싸이즈','핏이','핏','핏입니다',
      '핏감도','핏나','핏감이랑','핏처럼','핏입니당','핏이에요','핏이예요','핏이랑','핏이구요','핏이네요','핏감은',
      '핏입니','핏이라서','핏으로','옷핏','핏더','핏도','핏감','옷','후드','후드티','핏은','좋아요핏','요핏도',
      '좋아요핏도','상의','요핏','폼','폭','사이즈','후드','옷','품','전체','품도','몸통','옷핏','핏',
      '핏이구','핏이었','핏임','핏입',
      '핏입니','핏감','요핏','폼','후드핏','핏이','크기','상품','상의','핏더']
chongjang = ['길이','기장','총장','궁디','궁딩','엉덩이','밑','밑단','위아래','끝단','밑기장',
             '밑단이랑','기장이','기장입니','밑단','밑','기장감','요기장','총길이','총장','총기장',
            '길이감','끝','기장','길이']
shoulder = ['어깨핏','어께','어깨','너비','어깡','골격','바디','가로','넓이','등판','통',
            '어깨깡패','어깨라','어깨넓','어깨쪽','어께','몸집','어깨핏','어깨선','어깨라인','어깡',
            '어깨','통','몸통','골격','넓이','너비','등판']
chest = ['가슴','둘레','바디','가로','넓이','통','둘레','몸집','가슴팍','통','몸통']
arm = ['소매','팔도','손목','손','팔다리','팔목','팔','팔꿈치','팔이','요팔',
       '당팔','팔길이','팔다리','팔길','팔소매','손목','팔목','소매통','손','팔','소매','팔기장']
small = ['작습니다','짧았어요','짧습니다','짧으면','짧','작았어요','짧다고','쪼여서','작았으면',
      '작아도','작으면','작음','작아진','좁다고',
      '짧아요','짧게','짧다는','좁다는','작네','작기는','달라붙는','작은데','짧은데','짧음',
      '작긴','짧네요','작게','작아서','작은','짧고','짧아','좁은','타이트','짧았지만','붙어서','짧긴',
      '작은것','작아요','짧은','짧아서','짧지만','좁아','좁고','작네요','작아','쪼이는','좁아요','크롭느낌',
      '크롭하','작고','숏','쫄려','작다고','작다','좁아서','작','크롭','크롭된','미니','크롭해',
      '길었으면','컸으면','좁게','짧지' ,'작지','크롭이',
      '숏하','사이즈업하','크롭하','숏한','크롭하긴','크롭이','크롭해','크롭느낌','크롭된','사이즈업',
       '작','숏','짧긴','크롭','크롭한','타이트','크롭이라','작긴',
       '작','짧','좁']
big = ['큰',' 긴','커요','크고','크긴','넉넉하고','길어서','넉넉하게','작다는'
     '넉넉한','클','길고','큰데','크지','크지만','부해','크니깐','크더라구요','컸어도','넓음','큼직하게',
     '박시하네요','넉넉할','헐렁하게','큼직하니','컸음','큽니','긴데','길어','길게','넓어서','넓어','박시하고','컸어요',
     '박시','크다고','크다는','넓게','넓고','커용','길다는','크지도','큼','크다','길었어요','넉넉함','큰데도','덮는데',
     '덮이고','컸고','벙벙해서','크더라고요','컸을','크거나','크더라','컸었는데','컸어용','컸습니다','덮네요',
     '큼지막해서','덥는','박시하게','크네요','클까','넉넉하구','박시하','컸는데','넉넉하네요','크게','크니까','넉넉한데','길어요',
     '컸지만','헐렁한','크네','넉넉해','펑퍼짐한','덮어용','박시하니','헐렁해','넉넉하지''넉넉하구요','커','큽니다','넉넉하니',
     '길지','박시한데','덮어요','길긴','넉넉해요','넓은','덮히는', '넓긴','덮어줘서','덮은','넉넉해서','큼직해서','기네','낙낙',
     '흘러내리는','덮어서','벙벙한','커서','큰것','박시핏','오퍼핏','낙낙해','여유핏','널널함','덮어','넉넉합니다','크면',
     '덮고','가오리','루즈핏','펑퍼짐','길었지만','헐렁하고','남아요','아방핏','박스핏','널널','오바핏','덮습니다','넉넉히',
     '헐렁해서','접어도','접어','접어서','널','넓어요','덮고도','오버','와이드','낭낭','헐렁','접으면','짧았으면','접고',
     '버핏','접어야','잡아먹힌','수선해서','오버핏','오버사이즈','오버사이징해서','낙낙하다','낙낙한데','낙낙해요','오버핏이예요','오버핏입니당',
       '요박시하','컸어용','낙낙합니','어벙벙하','와이드핏','크니깐','크네요옷','널널해요','오버핏나와요','널럴하',
       '요넉넉하','엄청큼','여유핏','오버느낌','아방핏','오버핏은','넓긴','넉넉함','입니다오버핏','큽니다오버핏',
       '큽니다제','큽니당','오버해요','요박시한','헐렁하긴','헐렁거리','아방방한','아방방','박시합니','박시했다',
       '빅사이즈','받았습니다오버핏','빡시하','루즈핏으로','낭낭한','오버핏이긴','박스핏','널널함','어벙벙',
       '요넉넉한','오퍼핏','오버핏이구요','낙낙해','낭낭해서','넉넉하구','넉넉한데','아방해','오버핏이라서',
       '낭낭하','박시해','커용','넉넉해','펑퍼짐','박시하지','널널','좋아요오버핏','어벙벙한','널널하','오바핏',
       '오버하지','오버핏하','박시합니다','오버핏되','박시함','버핏','박시한','낙낙하니','박시해요','크네용',
       '낙낙해서','큼직해서','루즈해','느슨','널널합니다','아방아','아방아방','루즈핏이','박시하','세미오버',
       '세미오버핏','루즈해서','오버핏이라','루즈핏이라','오버핏이네요','오버핏입니','어벙','오버해','박시한데',
       '아방','오버핏이에요','예뻐요오버핏','큼지막','박시핏','오버해서','루즈한','오버핏인데','넉넉하구요','박시할',
       '큼직','오버한','오버핏이','낙낙하','아방하','롱','오버핏으로','낙낙한','오버핏입니다','박시해서','와이드',
       '요오버핏','넉넉해서','큽니','오버','오버하','루즈','루즈핏','박시','넉넉','크긴','아방한','널널한',
       '헐렁','길긴','와이드하','아방해서','세미오버사이즈',             
        '널','덮','넓','벙벙하','덮이','덮히','커다랗','덥히','넉넉해지','뒤집'             
       '넉넉']
total     = sizedict_nodup(total)
chongjang = sizedict_nodup(chongjang)
shoulder  = sizedict_nodup(shoulder)
arm       = sizedict_nodup(arm)
chest     = sizedict_nodup(chest)
small     = sizedict_nodup(small)
big       = sizedict_nodup(big)

