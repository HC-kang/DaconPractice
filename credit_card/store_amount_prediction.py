#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt


# In[2]:


# import warnings
# warnings.filterwarnings('ignore')


# In[3]:


import os
os.chdir('C:\dacon_data')


# In[4]:


train = pd.read_csv('funda_train.csv')


# In[5]:


train.head()


# 데이터의 형태 확인

# In[6]:


plt.figure(figsize = (13, 4))
plt.bar(train.columns, train.isnull().sum())
plt.xticks(rotation = 45)


# 확인결과 region 과 type_of_business 필드의 다수가 결측치이고, 대치도 불가능한 쓸모없는 정보들임.

# In[7]:


train = train.drop(['region', 'type_of_business'], axis = 1)
train.head()


# In[8]:


plt.figure(figsize=(8, 4))
sns.boxplot(train['amount'])


# 생긴 건 둘째치고, amount(금액) 에 음수값이 다량 존재함. 이는 환불 등으로 인해 발생한 것으로, 매출 측정에 방해가 되는 노이즈임.

# In[9]:


train[train['amount']<0].head()


# 

# In[10]:


# 거래일과 거래시간을 합친 변수 생성
train['datetime'] = pd.to_datetime(train.transacted_date + ' '+
                                   train.transacted_time, format = '%Y-%m-%d %H:%M:%S')
# 환불 거래를 제거하는 함수 정의
def remove_refund(df):
    refund = df[df['amount']<0] # 매출액 음숫값 데이터를 추출
    non_refund = df[df['amount']>0] # 매출액 양숫값 데이터를 추출
    removed_data = pd.DataFrame()
    
    for i in tqdm(df.store_id.unique()):
        # 매출액이 양숫값인 데이터를 상점별로 나누기
        divided_data = non_refund[non_refund['store_id']==i]
        # 매출액이 음숫값인 데이터를 상점별로 나누기
        divided_data2 = refund[refund['store_id']==i]
        
        for neg in divided_data2.to_records()[:]: # 환불 데이터를 차례대로 검사합니다.
            refund_store = neg['store_id']
            refund_id = neg['card_id'] # 환불 카드 아이디를 추출
            refund_datetime = neg['datetime'] # 환불 시간을 추출
            refund_amount = abs(neg['amount']) # 매출 음숫값의 절댓값을 구합니다.
            
            # 환불 시간 이전의 데이터 중 카드 아이디와 환불액이 같은 후보리스트 뽑기
            refund_pay_list = divided_data[divided_data['datetime']<=refund_datetime]
            refund_pay_list = refund_pay_list[refund_pay_list['card_id']==refund_id]
            refund_pay_list = refund_pay_list[refund_pay_list['amount']==refund_amount]

            # 후보 리스트가 있으면 카드 아이디, 환불액이 같으면서 가장 최근시간을 제거
            if len(refund_pay_list) != 0:
                # 가장 최근시간 구하기
                refund_datetime = max(refund_pay_list['datetime'])
                # 가장 최근 시간
                noise_list = divided_data[divided_data['datetime']==refund_datetime]
                # 환불 카드 아이디
                noise_list = noise_list[noise_list['card_id']==refund_id]
                # 환불액
                noise_list = noise_list[noise_list['amount']==refund_amount]
                # 인덱스를 통해 제거
                divided_data = divided_data.drop(index = noise_list.index)

        # 제거한 데이터를 데이터프레임에 추가
        removed_data = pd.concat([removed_data, divided_data], axis = 0)
    return removed_data


# In[11]:


positive_data = remove_refund(train)
plt.figure(figsize=(8, 4))
sns.boxplot(positive_data['amount'])


# In[12]:


# 5개의 행을 출력
positive_data.head()


# In[13]:


# 월 단위 다운샘플링 함수 정의
def month_resampling(df):
    new_data = pd.DataFrame()
    # 연도와 월을 합친 변수를 생성
    df['year_month'] = df['transacted_date'].str.slice(stop = 7)
    # 데이터의 전체 기간을 추출
    year_month = df['year_month'].drop_duplicates()
    # 상점 아이디별로 월 단위 매출액 총합 구하기
    downsampling_data = df.groupby(['store_id', 'year_month']).amount.sum()
    downsampling_data = pd.DataFrame(downsampling_data)
    downsampling_data = downsampling_data.reset_index(drop = False, inplace = False)
    
    for i in tqdm(df.store_id.unique()):
        # 상점별로 데이터 처리
        store = downsampling_data[downsampling_data['store_id']==i]
        # 각 상점의 처음 매출이 발생한 월을 구하기
        start_time = min(store['year_month'])
        # 모든 상점을 전체 기간 데이터로 만들기
        store = store.merge(year_month, how = 'outer')
        # 데이터를 시간순으로 정렬
        store = store.sort_values(by = ['year_month'], axis = 0, ascending = True)
        # 매출이 발생하지 않는 월은 2로 채우기
        store['amount'] = store['amount'].fillna(2)
        # 상점 아이디 결측치 채우기
        store['store_id'] = store['store_id'].fillna(i)
        # 처음 매출이 발생한 월 이후만 뽑기
        store = store[store['year_month']>=start_time]
        
        new_data = pd.concat([new_data, store], axis = 0)
    return new_data


# In[14]:


# 환불 제거 데이터를 월 단위로 다운샘플링
resampling_data = month_resampling(positive_data)
resampling_data['store_id'] = resampling_data['store_id'].astype(int)
resampling_data


# In[15]:


print(type(resampling_data))


# In[16]:


# 데이터프레임을 Series로 변환하는 함수
def time_series(df, i):
    # 상점별로 데이터 뽑기
    store = df[df['store_id']==i]
    # 날짜 지정 범위는 영어 시작 월부터 2019년 3월 전 영업마감일까지.
    index = pd.date_range(min(store['year_month']), '2019-03', freq = 'BM')
    # 시리즈 객체로 변환
    ts = pd.Series(store['amount'].values, index = index)
    return ts


# In[17]:


store_0 = time_series(resampling_data, 0)
store_0


# In[18]:


# 데이터 타입을 출력
store_1 = time_series(resampling_data, 1)
print(type(store_1))


# In[19]:


# 상점 아이디가 2번인 데이터를 시리즈 객체로 변환
store_2 = time_series(resampling_data, 2)
store_2.plot()


# In[20]:


# 상점 아이디가 257번인 데이터를 시리즈 객체로 데이터 출력
store_257 = time_series(resampling_data, 257)
store_257


# In[21]:


# 시계열 그래프 그리기
store_plot_257 = store_257.plot()
fig = store_plot_257.get_figure()
fig.set_size_inches(13.5, 5)


# In[22]:


# 상점 아이디가 2096번인 데이터를 시리즈 객체로 데이터 출력
store_2096 = time_series(resampling_data, 2096)
store_2096


# In[23]:


# 상점 아이디가 2096번인 상점의 시계열 그래프
store_plot_2096 = store_2096.plot()
fig = store_plot_2096.get_figure()
fig.set_size_inches(13.5, 5)


# In[24]:


# 상점 아이디가 335번인 상점의 시계열 그래프
store_335 = time_series(resampling_data, 335)
store_plot_335 = store_335.plot()
fig = store_plot_335.get_figure()
fig.set_size_inches(13.5, 5)


# In[25]:


# 상점 아이디가 510번인 상점의 시계열 그래프
store_510 = time_series(resampling_data, 510)
store_plot_510 = store_510.plot()
fig = store_plot_510.get_figure()
fig.set_size_inches(13.5,5)


# In[26]:


# 상점 아이디가 111번인 데이터를 시리즈 객체로 출력
store_111 = time_series(resampling_data, 111)
store_111


# In[27]:


# 상점 아이디가 111번인 상점의 시계열 그래프
store_plot_111 = store_111.plot()
fig = store_plot_111.get_figure()
fig.set_size_inches(13.5, 5)


# In[28]:


# 상점 아이디가 279번인 데이터 시리즈 객체로 데이터 출력
store_279 = time_series(resampling_data, 279)
store_279


# In[29]:


# 상점 아이디가 279인 상점의 시계열 그래프
store_plot_279 = store_279.plot()
fig = store_plot_279.get_figure()
fig.set_size_inches(13.5, 5)


# In[30]:


# 상점 아이디가 0번인 상점의 시계열 그래프
store_0 = time_series(resampling_data, 0)
store_plot_0 = store_0.plot()
fig = store_plot_0.get_figure()
fig.set_size_inches(13.5, 5)


# In[31]:


# 상점 아이디가 257번인 상점의 시계열 그래프
store_257 = time_series(resampling_data, 257)
store_plot_257 = store_257.plot()
fig = store_plot_257.get_figure()
fig.set_size_inches(13.5, 5)


# 매출이 없던 795번 상점 - 망함

# In[49]:


# 상점 아이디가 795번인 상점의 시계열 그래프
store_795 = time_series(resampling_data, 795)
store_plot_795 = store_795.plot()
fig = store_plot_795.get_figure()
fig.set_size_inches(13.5, 5)


# In[32]:


# pmdarima 패키지에 있는 ADFTest 클래스를 임포트
from pmdarima.arima import ADFTest
    
# 상점 아이디가 0번인 데이터를 시리즈 객체로 변환
store_0 = time_series(resampling_data, 0)
# ADFTest 시행
p_val, should_diff = ADFTest().should_diff(store_0)
print('p_val : %f, should_diff : %s'%(p_val, should_diff))


# In[33]:


# 상점 아이디가 257번인 데이터를 시리즈 객체로 반환
store_257 = time_series(resampling_data, 257)
# ADFTest 시행
p_val, should_diff = ADFTest().should_diff(store_257)
print('p_val : %f, should_diff : %s'%(p_val, should_diff))


# In[39]:


# ARIMA 모델의 차분 여부를 결정하기 위한 단위근 검정
def adf_test(y):
    return ADFTest().should_diff(y)[0]

# 전체 상점 adf_test p-value 값을 리스트에 저장해 boxplot으로 표시
adf_p = []
count = 0
skipped = []

for i in tqdm(resampling_data['store_id'].unique()):
    ts = time_series(resampling_data,i)
    try:
        p_val = adf_test(ts)
        if p_val < 0.05:
            count += 1
        adf_p.append(p_val)
    except:
        skipped.append(i)

plt.figure(figsize=(8, 4))
sns.boxplot(adf_p)

# # 연습
# def adf_test(y):
#     return ADFTest(),should_diff(y)[0]

# adf_p = []
# count = 0
# skipped = []

# for i in tqdm(resampling_data['store'].unique()):
#     ts = time_series(resampling_data, i)
#     try:
#         p_val = adf_test(ts)
#         if p_val < 0.05:
#             count += 1
#         adf_p.append(p_val)
#     except:
#         skipped.append(i)

# plt. figure(figsize = (8, 4))
# sns.boxplot(adf_p)

# In[35]:


# p-value가 0.05보다 작은 상점의 개수
print(count)


# In[36]:


skipped


# In[37]:


# ADF-Test 오류 상점 개수
if skipped:
    print(f"WarningCount: {len(skipped)}, store_id_list:{skipped}")


# 모델 구축과 검증

# In[38]:


from rpy2.robjects.packages import importr # rpy2 내의 패키지를 불러올 importr클래스

utils = importr('utils') # utils 패키지를 임포트
utils.install_packages('forecast') # r의 forecast 패키지 설치
utils.install_packages('forecastHybrid') # r의 forecastHybrid 패키지 설치


# In[40]:


import rpy2.robjects as robjects # r 함수를 파이썬에서 사용 가능하게 변환하는 모듈
from rpy2.robjects import pandas2ri # 파이썬 자료형과 R 자료형의 호환을 도와주는 모듈

# pandas2ri를 활성화 
pandas2ri.activate()

auto_arima = """
    function(ts){
        library(forecast) # forecast 패키지 로드
        d_params = ndiffs(ts) # 시계열 자료의 차분 횟수 계산
        model = auto.arima(ts, max.p=2, d=d_params) # auto.arima 모델 생성
        forecasted_data = forecast(model, h=3) # 이후 3개월(h=3)을 예측
        out_df = data.frame(forecasted_data$mean) # 예측값을 R의 데이터프레임으로 변환
        colnames(out_df) = c('amount') # amount라는 열로 이름을 지정
        out_df
    }
"""
# r() 함수로 r 자료형을 파이썬에서 사용 가능
auto_arima = robjects.r(auto_arima)
ts = robjects.r('ts')# r 자료형 time series 자료형으로 만들어주는 함수
c = robjects.r('c') # r 자료형 벡터를 만들어주는 함수

store_0 = resampling_data[resampling_data['store_id']==0]
start_year = int(min(store_0['year_month'])[:4]) # 영업 시작 년도
start_month = int(min(store_0['year_month'])[5:]) # 영업 시작 월
    
# R의 ts 함수로 r의 time series 자료형으로 변환
train = ts(store_0['amount'], start=c(start_year, start_month), frequency=12) 

#ensemble model
forecast = auto_arima(train)
np.sum(pandas2ri.ri2py(forecast).values) # 3개월 매출을 합산


# In[41]:


import rpy2.robjects as robjects # r 함수를 파이썬에서 사용 가능하게 변환하는 모듈
from rpy2.robjects import pandas2ri # 파이썬 자료형과 R 자료형의 호환을 도와주는 모듈

# pandas2ri를 활성화 
pandas2ri.activate()

auto_arima = """
    function(ts){
        library(forecast) # forecast 패키지 로드
        d_params = ndiffs(ts) # 시계열 자료의 차분 횟수 계산
        model = auto.arima(ts, max.p=2, d=d_params) # auto.arima 모델 생성
        forecasted_data = forecast(model, h=3) # 이후 3개월(h=3)을 예측
        out_df = data.frame(forecasted_data$mean) # 예측값을 R의 데이터프레임으로 변환
        colnames(out_df) = c('amount') # amount라는 열로 이름을 지정
        out_df
    }
"""
# r() 함수로 r 자료형을 파이썬에서 사용 가능
auto_arima = robjects.r(auto_arima)# str 형식으로 정의된 auto_arima
ts = robjects.r('ts')# r 자료형 time series 자료형으로 만들어주는 함수
c = robjects.r('c') # r 자료형 벡터를 만들어주는 함수

final_pred = []

for i in tqdm(resampling_data.store_id.unique()):
    store = resampling_data[resampling_data['store_id']==i]
    start_year = int(min(store['year_month'])[:4]) ## 영업 시작 년도
    start_month = int(min(store['year_month'])[5:]) ## 영업 시작 월
    # R의 ts 함수로 time series 데이터로 변환
    train = ts(store['amount'], start=c(start_year, start_month), frequency=12) 
    # 자동회귀누적이동평균 model
    forecast = auto_arima(train)
    # 3개월 매출을 합산, final_pred에 추가
    final_pred.append(np.sum(pandas2ri.ri2py(forecast).values))


# In[42]:


submission = pd.read_csv('./submission.csv')
submission['amount'] = final_pred
submission.to_csv('submission.csv', index=False)
submission


# In[43]:


import rpy2.robjects as robjects # r 함수를 파이썬에서 사용 가능하게 변환하는 모듈
from rpy2.robjects import pandas2ri # 파이썬 자료형과 R 자료형의 호환을 도와주는 모듈

# pandas2ri를 활성화 
pandas2ri.activate()

ets = """
    function(ts){
        library(forecast) # forecast 패키지 로드
        model = ets(ts) # AIC가 낮은 지수평활 모델을 찾음 
        forecasted_data = forecast(model, h=3) # 이후 3개월(h=3)을 예측
        out_df = data.frame(forecasted_data$mean) # 예측값을 R의 데이터프레임으로 변환
        colnames(out_df) = c('amount') # amount라는 열로 이름을 지정
        out_df
    }
"""
# r() 함수로 r 자료형을 파이썬에서 사용 가능
ets = robjects.r(ets)# str 형식으로 정의된 ets
ts = robjects.r('ts')# r 자료형 time series 자료형으로 만들어주는 함수
c = robjects.r('c') # r 자료형 벡터를 만들어주는 함수

final_pred = []

for i in tqdm(resampling_data.store_id.unique()):
    store = resampling_data[resampling_data['store_id']==i]
    start_year = int(min(store['year_month'])[:4]) # 영업 시작 년도
    start_month = int(min(store['year_month'])[5:]) # 영업 시작 월
    # R의 ts 함수로 time series 데이터로 변환
    train = ts(store['amount'], start=c(start_year, start_month), frequency=12) 
    # 지수평활법l
    forecast = ets(train)
    # 3개월 매출을 합산, final_pred에 추가
    final_pred.append(np.sum(pandas2ri.ri2py(forecast).values))


# In[45]:


submission = pd.read_csv('./submission.csv')
submission['amount'] = final_pred
submission.to_csv('submission.csv', index=False)


# In[46]:


from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

store_0 = time_series(resampling_data, 0)
# STL 분해
stl = seasonal_decompose(store_0.values, freq=12)
stl.plot()
plt.show()


# In[47]:


import rpy2.robjects as robjects # r 함수를 파이썬에서 사용 가능하게 변환하는 모듈
from rpy2.robjects import pandas2ri # 파이썬 자료형과 R 자료형의 호환을 도와주는 모듈

# pandas2ri를 활성화 
pandas2ri.activate()
stlm = """
    function(ts){
        library(forecast) # forecast 패키지 로드
        model = stlm(ts, s.window="periodic", method='ets') # STL 분해 후 지수평활법을 통한 예측 
        forecasted_data = forecast(model, h=3) # 이후 3개월(h=3)을 예측
        out_df = data.frame(forecasted_data$mean) # 예측값을 R의 데이터프레임으로 변환
        colnames(out_df) = c('amount') # amount라는 열로 이름을 지정
        out_df
    }
"""
ets = """
    function(ts){
        library(forecast) # forecast 패키지 로드
        model = ets(ts) # AIC가 낮은 지수평활 모델을 찾음 
        forecasted_data = forecast(model, h=3) # 이후 3개월(h=3)을 예측
        out_df = data.frame(forecasted_data$mean) # 예측값을 R의 데이터프레임으로 변환
        colnames(out_df) = c('amount') # amount라는 열로 이름을 지정
        out_df
    }
"""
# r() 함수로 r을 파이썬에서 사용 가능
stlm = robjects.r(stlm)# str 형식으로 정의된 stlm
ets = robjects.r(ets)# str 형식으로 정의된 ets
ts = robjects.r('ts')# r 자료형 time series 자료형으로 만들어주는 함수
c = robjects.r('c') # r 자료형 벡터를 만들어주는 함수

final_pred = []

for i in tqdm(resampling_data.store_id.unique()):
    store = resampling_data[resampling_data['store_id']==i]
    data_len = len(store)
    
    start_year = int(min(store['year_month'])[:4]) # 영업 시작 년도
    start_month = int(min(store['year_month'])[5:]) # 영업 시작 월
    # R의 ts 함수로 time series 데이터로 변환
    train = ts(store['amount'], start=c(start_year, start_month), frequency=12) 
    # STL 분해를 적용한 지수평활 model
    if data_len > 24:
        forecast = stlm(train)
    # 지수평활 model
    else:
        forecast = ets(train)    # 3개월 매출을 합산, final_pred에 추가
    final_pred.append(np.sum(pandas2ri.ri2py(forecast).values))


# In[48]:


submission = pd.read_csv('./submission.csv')
submission['amount'] = final_pred
submission.to_csv('submission.csv', index=False)
submission


# 4.5 성능 향상을 위한 방법

# 4.5.1 상점 매출액의 로그 정규화

# In[51]:


import rpy2.robjects as robjects # r 함수를 파이썬에서 사용 가능하게 변환하는 모듈
from rpy2.robjects import pandas2ri # 파이썬 자료형과 R 자료형의 호환을 도와주는 모듈
import numpy as np

# pandas2ri를 활성화 
pandas2ri.activate()

auto_arima = """
    function(ts){
        library(forecast) # forecast 패키지 로드
        d_params = ndiffs(ts) # 시계열 자료의 차분 횟수 계산
        model = auto.arima(ts, max.p=2, d=d_params) # auto.arima 모델 생성
        forecasted_data = forecast(model, h=3) # 이후 3개월(h=3)을 예측
        out_df = data.frame(forecasted_data$mean) # 예측값을 R의 데이터프레임으로 변환
        colnames(out_df) = c('amount') # amount라는 열로 이름을 지정
        out_df
    }
"""

# r() 함수로 r 자료형을 파이썬에서 사용 가능
auto_arima = robjects.r(auto_arima)
ts = robjects.r('ts')# r 자료형 time series 자료형으로 만들어주는 함수
c = robjects.r('c') # r 자료형 벡터를 만들어주는 함수
log = robjects.r('log')# 로그 변환 함수
exp = robjects.r('exp')# 로그 역변환 함수

# 0번 상점 추출
store_0 = resampling_data[resampling_data['store_id']==0]
start_year = int(min(store_0['year_month'])[:4]) # 영업 시작 년도
start_month = int(min(store_0['year_month'])[5:]) # 영업 시작 월

# train, test 분리
train = store_0[store_0.index <= len(store_0)-4]
test = store_0[store_0.index > len(store_0)-4]

# R의 ts 함수로 r의 time series 자료형으로 변환
train_log = ts(log(train['amount']), start=c(start_year, start_month), frequency=12) # log 정규화 
train = ts(train['amount'], start=c(start_year, start_month), frequency=12) # log 정규화를 하지 않음

# model arima
forecast_log = auto_arima(train_log)
forecast = auto_arima(train)

# pred
pred_log = np.sum(pandas2ri.ri2py(exp(forecast_log)).values) #로그 역변환 후 3개월 합산
pred = np.sum(pandas2ri.ri2py(forecast).values) #3개월 매출을 합산

# test(2018-12~2019-02)
test = np.sum(test['amount'])

# mae
print('log-regularization mae: ', abs(test-pred_log))
print('mae:', abs(test-pred))


# In[52]:


# 매출 변동 계수를 구하는 함수
def coefficient_variation(df, i):
    cv_data = df.groupby(['store_id']).amount.std()/df.groupby(['store_id']).amount.mean()
    cv = cv_data[i]
    return cv


# In[53]:


import rpy2.robjects as robjects # r 함수를 파이썬에서 사용 가능하게 변환하는 모듈
from rpy2.robjects import pandas2ri # 파이썬 자료형과 R 자료형의 호환을 도와주는 모듈
import numpy as np

# pandas2ri를 활성화 
pandas2ri.activate()

ets = """
    function(ts){
        library(forecast) # forecast 패키지 로드
        model = ets(ts) # AIC가 낮은 지수평활 모델을 찾음 
        forecasted_data = forecast(model, h=3) # 이후 3개월(h=3)을 예측
        out_df = data.frame(forecasted_data$mean) # 예측값을 R의 데이터프레임으로 변환
        colnames(out_df) = c('amount') # amount라는 열로 이름을 지정
        out_df
    }
"""

# r() 함수로 r 자료형을 파이썬에서 사용 가능
ets = robjects.r(ets)
ts = robjects.r('ts') # r 자료형 time series 자료형으로 만들어주는 함수
c = robjects.r('c') # r 자료형 벡터를 만들어주는 함수
log = robjects.r('log') # 로그 변환 함수
exp = robjects.r('exp')# 로그 역변환 함수

final_pred = []

for i in tqdm(resampling_data.store_id.unique()):
    store = resampling_data[resampling_data['store_id']==i]
    start_year = int(min(store['year_month'])[:4]) # 영업 시작 년도
    start_month = int(min(store['year_month'])[5:]) # 영업 시작 월
    
    cv = coefficient_variation(resampling_data, i)
    # 매출액 변동 계수가 0.3 미만인 경우만 log를 씌움
    if cv < 0.3:
        train_log = ts(log(store['amount']), start=c(start_year,start_month), frequency=12) 
        # ets model
        forecast_log = ets(train_log)
        final_pred.append(np.sum(pandas2ri.ri2py(exp(forecast_log)).values))
    # 매출액 변동 계수가 0.3 이상인 경우
    else:
        train = ts(store['amount'], start=c(start_year,start_month), frequency=12)
        # 지수평활법
        forecast = ets(train)
        final_pred.append(np.sum(pandas2ri.ri2py(forecast).values))


# In[54]:


submission = pd.read_csv('./submission.csv')
submission['amount'] = final_pred
submission.to_csv('submission.csv', index=False)
submission


# 4.5.2. 파이썬에서 R 시계열 패키지 forecastHybrid를 통한 앙상블

# In[55]:


import rpy2.robjects as robjects # r 함수를 파이썬에서 사용 가능하게 변환하는 모듈
from rpy2.robjects import pandas2ri # 파이썬 자료형과 R 자료형의 호환을 도와주는 모듈
import numpy as np

# pandas2ri를 활성화 
pandas2ri.activate()

hybridModel = """
    function(ts){
        library(forecast)
        library(forecastHybrid)
        d_params=ndiffs(ts)
        hb_mdl<-hybridModel(ts, models="aes", # auto_arima, ets, stlm
                        a.arg=list(max.p=2, d=d_params), # auto_arima parameter
                        weight="equal") # 가중치를 동일하게 줌(평균)
        forecasted_data<-forecast(hb_mdl, h=3) # 이후 3개월(h=3)을 예측
        outdf<-data.frame(forecasted_data$mean)
        colnames(outdf)<-c('amount')
        outdf
    }
""" 

# r() 함수로 r 자료형을 파이썬에서 사용 가능
hybridModel = robjects.r(hybridModel)
ts = robjects.r('ts') # r 자료형 time series 자료형으로 만들어주는 함수
c = robjects.r('c') # r 자료형 벡터를 만들어주는 함수
log = robjects.r('log') # 로그 변환 함수
exp = robjects.r('exp')# 로그 역변환 함수

final_pred = []

for i in tqdm(resampling_data.store_id.unique()):
    store = resampling_data[resampling_data['store_id']==i]
    start_year = int(min(store['year_month'])[:4]) # 영업 시작 년도
    start_month = int(min(store['year_month'])[5:]) # 영업 시작 월
    
    cv = coefficient_variation(resampling_data, i)
    # 매출액 변동 계수가 0.3 미만인 경우만 log를 씌움
    if cv < 0.3:
        train_log = ts(log(store['amount']), start=c(start_year,start_month), frequency=12) 
        # 앙상블 예측
        forecast_log = hybridModel(train_log)
        final_pred.append(np.sum(pandas2ri.ri2py(exp(forecast_log)).values)) 
    # 매출액 변동 계수가 0.3 이상인 경우
    else:
        train = ts(store['amount'], start=c(start_year,start_month), frequency=12)
        # 앙상블 예측
        forecast = hybridModel(train)
        final_pred.append(np.sum(pandas2ri.ri2py(forecast).values))


# In[56]:


submission = pd.read_csv('./submission.csv')
submission['amount'] = final_pred
submission.to_csv('submission.csv', index=False)
submission


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




