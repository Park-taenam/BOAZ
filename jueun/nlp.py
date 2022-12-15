import nltk
from nltk.stem.snowball import SnowballStemmer;from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer; from sklearn import linear_model
from sklearn.model_selection import KFold
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from matplotlib import pyplot as plt
# 라이브러리 로드 
from sklearn.feature_extraction.text import HashingVectorizer
from konlpy.tag import *
from konlpy.utils import pprint
from sklearn.pipeline import Pipeline
import statsmodels.api as sm

#import lda
import pandas as pd
import numpy as np
import re
import os
import sys
import json

from konlpy.tag import Kkma, Komoran, Okt, Mecab
import soynlp
import re
import collections
import itertools
import requests
import csv
import time
import math
import operator

from tqdm import tqdm
from collections import Counter
from collections import defaultdict
from pandas import read_table
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

############################################################################

# 한국어 nlp
from konlpy.tag import *   # 모든 형태소분석기 import 하기
okt = Okt()

# 영어 nlp 관련
from nltk.tokenize import word_tokenize

# 시각화 관련
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
from PIL import Image

# 기타
from tqdm import tqdm_notebook, tqdm   # for문 진행상황 눈으로 확인 (loading bar)
import datetime