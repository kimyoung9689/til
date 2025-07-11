# 0. 초기 설정 및 필수 라이브러리 설치

!pip install eli5==0.13.0
!pip install fuzzywuzzy
!pip install python-Levenshtein
!pip install haversine

# 한글 폰트 사용을 위한 라이브러리
!apt-get update -qq
!apt-get install -y fonts-nanum -qq

# --- 1. 필요한 라이브러리 모두 임포트 ---

# visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 한글 폰트 설정 (오류 처리 제외)
fe = fm.FontEntry(
    fname=r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf', # ttf 파일이 저장되어 있는 경로
    name='NanumBarunGothic') # 이 폰트의 원하는 이름 설정
fm.fontManager.ttflist.insert(0, fe) # Matplotlib에 폰트 추가
plt.rcParams.update({'font.size': 10, 'font.family': 'NanumBarunGothic'}) # 폰트 설정
plt.rc('font', family='NanumBarunGothic')


# utils
import pandas as pd
import numpy as np
import re # <-- 추가: 정규표현식 (아파트명 클리닝에 사용)
from tqdm import tqdm
import pickle
import warnings;warnings.filterwarnings('ignore')
import os
from IPython.display import display
import time
import requests


# Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from haversine import haversine, Unit

import eli5
from eli5.sklearn import PermutationImportance

# Fuzzy Matching 라이브러리 (아파트명 매칭에 필요해서 추가)
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# --- 3. 데이터 불러오기 ---

# 기본 데이터 불러오기
train_path = 'data/train.csv'
test_path  = 'data/test.csv'
bus_path = 'data/bus_feature.csv'
subway_path = 'data/subway_feature.csv'

# 대회 제출 양식
submission_path = 'data/sample_submission.csv'

# 외부 데이터 불러오기
seoul_apt_info_path = 'data/seoul_apt_info.csv'
seoul_apt_info_df = pd.read_csv(seoul_apt_info_path, encoding='cp949')
interest_rate_path = 'data/한은 기준금리_2007_2023.csv'

dt = pd.read_csv(train_path)     # 훈련 데이터
dt_test = pd.read_csv(test_path) # 테스트 데이터

print('Train data shape : ', dt.shape, 'Test data shape : ', dt_test.shape)

# Pandas가 모든 컬럼을 표시하도록 설정
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Train과 Test data를 살펴보기
display(dt.head(1))
display(dt_test.head(1))      # 부동산 실거래가(=Target) column이 제외된 모습

# train/test 구분을 위한 칼럼 생성
dt['is_test'] = 0
dt_test['is_test'] = 1
concat = pd.concat([dt, dt_test]) # 하나의 데이터로 만듦(둘이 비슷하니까 먼저 전처리)

# 모든 컬럼을 표시하도록 설정
pd.set_option('display.max_columns', None)
# 화면 폭을 넓게 설정 (모든 컬럼이 한 줄에 보이도록)
pd.set_option('display.width', 1000)

display(concat.head(1))

# 칼럼 이름 쉽게 변경
concat = concat.rename(columns={'전용면적(㎡)':'전용면적'})

# 모든 컬럼을 표시하도록 설정 (이전에 했겠지만 혹시 모르니 다시)
import pandas as pd # 혹시 모르니 다시 넣어줌
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 컬럼 이름에서 'k-' 접두사 제거
new_columns = []
for col in concat.columns:
    if col.startswith('k-'):
        new_columns.append(col[2:]) # 'k-' 다음부터의 문자열을 새 이름으로
    else:
        new_columns.append(col)

concat.columns = new_columns # 데이터프레임의 컬럼 이름을 새로운 리스트로 업데이트

# 이름이 변경되었는지 확인
display(concat.head(1))
print(concat.columns.tolist()) # 모든 컬럼 이름을 리스트로 출력해서 확인

# `seoul_apt_info_df` 데이터프레임의 첫 1행만 출력
display(seoul_apt_info_df.head(1))

# `seoul_apt_info_df` 데이터프레임의 모든 컬럼 이름을 리스트로 출력
print(seoul_apt_info_df.columns.tolist())




