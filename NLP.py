# %%

from tqdm import tqdm
# %%
#total,chongjang,shoulder,chest,arm ->keywords리스트


def get_keywords(keywords, df,keyword_column_name):
    start = time.time()
    review = df['review']
    df['new_column'] = str('')
    for i in tqdm(range(len(review))):
        
        keywords_search = []
        for j in keywords:
            if re.findall(j, review[i]):
                a = re.findall(j +'+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+',review[i]) #키워드 +단어
                aa = re.findall(j + ' '+'+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+',review[i])#키워드 +띄고 단어
                b = re.findall('[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+' + j ,review[i]) #단어 + 키워드
                bb = re.findall('[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+' + j ,review[i])#단어 + 띄고 키워드
                
                
                  #앞에 지만/데/고/보다 등이 들어있으면 키워드에 대한 수식어구가 아님! 
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
            df['new_column'][i] = keywords_o 
            
        else:
            df['new_column'][i] = '0'
    
    a = df.rename(columns = {'new_column':keyword_column_name})
    end = time.time()
    print("{:.5f} sec".format(end-start))
    return a[keyword_column_name]

def big_small(big,small,df,name_big_small):
    start = time.time()
    review = df[name_big_small+'_keyword']
    df[name_big_small + '_big'] = str('')
    df[name_big_small + '_small'] = str('')
    for i in tqdm(range(len(review))):
      
        big_search = []
        small_search = []
        for j in big:
            if re.findall(j, review[i]):
                a = re.findall(j,review[i]) #키워드
                big_search.extend(a)
               
        for j in small:
            if re.findall(j, review[i]):
                a = re.findall(j,review[i]) #키워드
                small_search.extend(a)
            
               
        
        if len(big_search) != 0:
            df[name_big_small + '_big'][i] = 1
            
        else:
            df[name_big_small + '_big'][i] = 0
        
        if len(small_search) != 0:
            df[name_big_small + '_small'][i] = 1
            
        else:
            df[name_big_small + '_small'][i] = 0
        
    end = time.time()
    print("{:.5f} sec".format(end-start))
    return df[[name_big_small + '_big',name_big_small + '_small']]












# %%
if __name__=="__main__":
    keyword_column_name = 'total_keyword'
    df_total            = get_keywords(total,df_before_nlp,keyword_column_name)
    keyword_column_name = 'chongjang_keyword'
    df_chongjang        = get_keywords(chongjang,df_before_nlp,keyword_column_name)
    keyword_column_name = 'shoulder_keyword'
    df_shoulder         = get_keywords(shoulder,df_before_nlp,keyword_column_name)
    keyword_column_name = 'chest_keyword'
    df_chest            = get_keywords(chest,df_before_nlp,keyword_column_name)
    keyword_column_name = 'arm_keyword'
    df_arm              = get_keywords(arm,df_before_nlp,keyword_column_name)

    df_keywords         = pd.concat([df_before_nlp,
                                    df_total,
                                    df_chongjang,
                                    df_chest,
                                    df_shoulder,
                                    df_arm],axis = 1)
    df_keywords =df_keywords.drop(['new_column'],axis = 1)
    df_keywords_origin =df_keywords.copy()
    
    
    
    
    name_big_small      = 'total'
    total_big_small     = big_small(big,small,df_keywords,name_big_small)
    name_big_small      = 'chongjang'
    chongjang_big_small = big_small(big,small,df_keywords,name_big_small)
    name_big_small      = 'arm'
    arm_big_small       = big_small(big,small,df_keywords,name_big_small)
    name_big_small      = 'chest'
    chest_big_small     = big_small(big,small,df_keywords,name_big_small)
    name_big_small      = 'shoulder'
    shoulder_big_small  = big_small(big,small,df_keywords,name_big_small)

    final_df            = df_keywords
    final_df_origin     = final_df.copy()
    
    
    final_df.to_pickle('data/crawlingdata_preprocess_review_done.pkl')