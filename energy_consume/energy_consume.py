import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import date
import matplotlib.pyplot as plt
from matplotlib import rc
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
import plotly.express as px
import os
import seaborn as sns
import tqdm
import gc

os.chdir('/Users/heechankang/projects/pythonworkspace/dacon_data/energy_consume')
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

train = pd.read_csv('train.csv', encoding='cp949')
test = pd.read_csv('test.csv', encoding='cp949')
sample_submission = pd.read_csv('sample_submission.csv', encoding='cp949')

train
train.tail()
train.shape #(122400, 10) 20.6.1~ 8.24
train.info()
train.describe()
train['date_time']=pd.to_datetime(train['date_time'])
train[train['num']==1]
train.isna().sum()
# 결측치 없음.

test.head(10)
test.tail()
test.shape #(10080, 9) 8.25~31 // 일주일
test.info()
test.describe()
test['date_time']=pd.to_datetime(test['date_time'])
test

plt.boxplot(train['전력사용량(kWh)'])

train.corr()
sns.heatmap(train.corr(), cmap='YlGn')
# 주요 변수는 온도, 바람??, 

plt.figure(figsize=(20,7))
plt.plot(train[train['num']==1]['전력사용량(kWh)'])

plt.figure(figsize=(20,7))
plt.plot(train[train['num']==2]['전력사용량(kWh)'])

plt.figure(figsize=(20,7))
plt.plot(train[train['num']==3]['전력사용량(kWh)'])

plt.figure(figsize=(20,10))
plt.plot(train[train['num']==10]['전력사용량(kWh)'])

plt.figure(figsize=(20,7))
plt.plot(train[train['num']==30]['전력사용량(kWh)'])

plt.figure(figsize=(20,7))
plt.plot(train[train['num']==60]['전력사용량(kWh)']) # 7월 1일
train[(train['num']==60) & (train['전력사용량(kWh)']<2000)]

# 빈껍데기
sample_submission.head()

data = train[['num','date_time', '전력사용량(kWh)']]

# 월별 나누기
data['month'] = data['date_time'].dt.month
data
# 월별 그래프
mean_month = data.groupby('month').mean()
fig=px.bar(mean_month, x=mean_month.index, y='전력사용량(kWh)')
fig.show()

# 요일별 나누기
data['weekday'] = data['date_time'].dt.weekday
data
# 요일별 그래프
mean_weekday = data.groupby('weekday').mean()
fig=px.bar(mean_weekday, x=mean_weekday.index, y='전력사용량(kWh)')
fig.show()

# 시간별 나누기
data['hour'] = data['date_time'].dt.hour
data
# 시간대별 그래프
mean_hour = data.groupby('hour').mean()
fig=px.bar(mean_hour, x=mean_hour.index, y='전력사용량(kWh)')
fig.show()

# 건물별 그래프
mean_hour = data.groupby('num').mean()
fig=px.bar(mean_hour, x=mean_hour.index, y='전력사용량(kWh)')
fig.show()

# ---*---*---* 데이터 가공 ---*---*---*

# 편의를 위해 변수명부터 바꿔주기
train_cols = list(train.columns)
train_cols
train.columns = ['num', 'date_time', 'energy', 'temp', 'wind', 'humid',
       'rain', 'sunshine', 'cooling', 'solar']

test_cols = list(test.columns)
test.columns = ['num', 'date_time', 'temp', 'wind', 'humid',
       'rain', 'sunshine', 'cooling', 'solar']

# 병합을 위해 column 만들어주기.
test['energy'] = 0

test = test[['num', 'date_time', 'energy', 'temp', 'wind', 'humid',
       'rain', 'sunshine', 'cooling', 'solar']]
test.isna().sum()
test



### test 에 있는 nan 값 잡아주기
def nan_feature(df, col_name):
    k = list(df.columns).index(col_name)
    for i in range(60):
        for j in range(0, 165, k):
            df.iloc[168*i+j+1,k] = (2*df.iloc[168*i+j,k]+df.iloc[168*i+j+3,k])/3
            df.iloc[168*i+j+2,k] = (df.iloc[168*i+j,k]+2*df.iloc[168*i+j+3,k])/3
    df[col_name].fillna(method='ffill', inplace = True)
    test[col_name] = test[col_name].round(1)
    print('결측치 확인:\n', df.isna().sum())
'''
아,, 인터폴레이트가 있는걸 몰랐다. 이걸 함수로 만들었네..
'''

# 1. temp
nan_feature(test, 'temp')

# 2. wind
nan_feature(test, 'wind')

# 3. humid
nan_feature(test, 'humid')

# 4. rain
# rain은 어차피 누적값이고 정규화로 나눠질테니 평균따위 필요없음.
test['rain'].fillna(method='ffill', inplace = True)

# 5. sunshine
test[test['sunshine']>=1]
nan_feature(test, 'sunshine')

# 6. cooling
test.describe()
test[test.cooling>=1]
test[test.num==1].head(10)
test[test.num==60].head(10)
test[test.num==60].head(10)

# 혹시 중간에 냉방기가 설치되었을 수 있으니 확인 - 다행히 그런건 없음
for i in range(60):
    print(test[test['num']==i]['cooling'].sum())

for i in range(60):
    if test.loc[i*168, 'cooling']==1:
        test.loc[i*168:i*168+167, 'cooling']=1
test['cooling'].fillna(0, inplace=True)
test.isna().sum()

# 7. solar 
for i in range(60):
    print(test[test['num']==i]['solar'].sum())
# 여기도 변화는 없음. 동일하게 채워주기
for i in range(60):
    if test.loc[i*168, 'solar']==1:
        test.loc[i*168:i*168+167, 'solar']=1
test['solar'].fillna(0, inplace=True)
test.isna().sum()

test
train
# 계산 및 전처리를 위해 임시적으로 test를 train에 concat
# 0825부터가 테스트셋임.
train = pd.concat([train, test], ignore_index=True)
train



# date_time 변수 쪼개기
train['month'] = train['date_time'].dt.month
train['day'] = train['date_time'].dt.day
train['hour'] = train['date_time'].dt.hour
train['weekday']=train['date_time'].dt.weekday
train['num_day'] = train['date_time'].dt.dayofyear - 152
train['isholy'] =  train['num_day'].apply(lambda x : (1 if x==78 else 0))
train['isweekend'] = train['weekday'].apply(lambda x : (1 if x==6 or x==7 else 0))
train[train['num_day']==78]
train
# 20년 6~8월 공휴일 현황
# 8월 17일

# 파생변수 생성 - 건물별 평균치
group = train.groupby(['num']).mean()
group.drop(['wind', 'rain', 'cooling', 'solar','month', 'day', 'hour','num_day','isweekend', 'isholy','weekday'], axis = 1, inplace=True)
group.columns = [str(col) +'_num' for col in group.columns]
group

train = pd.merge(train, group, on=['num'], how='left')


# 파생변수 생성 - 해보고싶은것
train['temp4'] = (273+train['temp'])**4
train['temp4'] = train['temp4'] / train['temp4'].max()
train[train['isweekend']==1]

# 파생변수 생성 - 건물별, 요일별 평균치
group = train.groupby(['num', 'weekday']).agg({'energy':'mean'})
group.columns = ['energy_num_weekday']
group

train = pd.merge(train, group, on=['num', 'weekday'], how = 'left')
train

# 파생변수 생성 - 불쾌지수
train['THI'] = 9/5*(train['temp'])-0.55*(1-(train['humid']/100))*(9/5*(train['temp'])-26)+32
train

# 파생변수 생성 - 일주일 이전까지 확인
# add lag feature
def day_lag_feature(df, lags, col):
    tmp = df[['num_day','num','hour',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['num_day','num','hour', col+'_day_lag_'+str(i)]
        shifted['num_day'] += i
        df = pd.merge(df, shifted, on=['num_day','num','hour'], how='left')
    return df

train = day_lag_feature(train, [1,2,3,4,5,6,7,14,21], 'energy')

train[train['num']==60].head(40)
train[train['num']==60].tail(40)

train

train.isna().sum()
train.columns
train.info()

# 파생변수 생성 - 3시간 이전까지 전력량 확인
# add lag feature
def hour_lag_feature(df, lags, col):
    tmp = df[['num_day','num','hour',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['num_day','num','hour', col+'_hour_lag_'+str(i)]
        shifted['hour'] += i
        df = pd.merge(df, shifted, on=['num_day','num','hour'], how='left')
    return df

train = hour_lag_feature(train, [1,2,3], 'energy')
train.temp4

###################
# 일단 대략적으로 끝
###################
def SMAPE(true, pred):
    v = 2 * abs(pred - true) / (abs(pred) + abs(true))
    output = np.mean(v) * 100
    return output


SMAPE(y_valid, y_pred)
# 이전 : 0.04320080970690549
# 이후 : 0.027917625681270052
# scaler : 8.545429389000786
# std_scaler : 21.307999985271227
# mm_scaler : 16.34143720271491
# ma_scaler : 8.780832133459596
# rb_scaler : 19.338556561555645

train.drop(['energy_hour_lag_1','energy_hour_lag_2','energy_hour_lag_3'], axis = 1, inplace=True)
train.drop(['energy_hour_lag_1','energy_hour_lag_2','energy_hour_lag_3','THI'], axis = 1, inplace=True)
train



# ---*---*---* 첫 번째 모델 ---*---*---*
# 불쾌지수 미적용
# train.to_pickle('full_train_nan.pkl')
train = pd.read_pickle('full_train_nan.pkl')
train.drop('date_time', axis = 1, inplace=True)
#train = train[train['num_day']>7]
# nan값 잡기
train.isna().sum()
train.fillna(0, inplace=True)

train[train['num_day']==78]
train[train['num_day']==85]

# 세트별 나눠주기
# ~0816
X_train = train[train.num_day < 78].drop(['energy'], axis=1)
y_train = train[train.num_day < 78]['energy']
# 0817 ~ 0824
X_valid = train[(train.num_day >= 78) & (train.num_day<86)].drop(['energy'], axis=1)
y_valid = train[(train.num_day >= 78) & (train.num_day<86)]['energy']
# 0825~
X_test = train[train.num_day >= 86].drop(['energy'], axis=1)
del train

# valid 없이 나누기
X_train = train[train.num_day < 86].drop(['energy'], axis=1)
y_train = train[train.num_day < 86]['energy']

X_test = train[train.num_day >= 86].drop(['energy'], axis=1)

# #########################
# from sklearn.preprocessing import StandardScaler

# std_scaler = StandardScaler()
# X_train_std = std_scaler.fit_transform(X_train)
# X_valid_std = std_scaler.fit_transform(X_valid)

# from sklearn.preprocessing import MinMaxScaler

# mm_scaler = MinMaxScaler()
# X_train_mm = mm_scaler.fit_transform(X_train)
# X_valid_mm = mm_scaler.fit_transform(X_valid)

####
# THI 제외
X_train_cat = X_train[['num','cooling', 'solar',
       'month', 'day', 'hour', 'weekday',  'isholy', 'isweekend',
        'group']]
X_train = X_train[['temp', 'wind', 'humid', 'rain', 'sunshine','num_day','energy_num', 'temp_num', 'humid_num', 'sunshine_num', 'temp4',
       'energy_num_weekday', 'energy_day_lag_1', 'energy_day_lag_2',
       'energy_day_lag_3', 'energy_day_lag_4', 'energy_day_lag_5',
       'energy_day_lag_6', 'energy_day_lag_7', 'energy_day_lag_14',
       'energy_day_lag_21']]

X_valid_cat = X_valid[['num','cooling', 'solar',
       'month', 'day', 'hour', 'weekday',  'isholy', 'isweekend',
        'group']]
X_valid = X_valid[['temp', 'wind', 'humid', 'rain', 'sunshine','num_day','energy_num', 'temp_num', 'humid_num', 'sunshine_num', 'temp4',
       'energy_num_weekday', 'energy_day_lag_1', 'energy_day_lag_2',
       'energy_day_lag_3', 'energy_day_lag_4', 'energy_day_lag_5',
       'energy_day_lag_6', 'energy_day_lag_7', 'energy_day_lag_14',
       'energy_day_lag_21']]

X_test_cat = X_test[['num','cooling', 'solar',
       'month', 'day', 'hour', 'weekday',  'isholy', 'isweekend',
        'group']]
X_test = X_test[['temp', 'wind', 'humid', 'rain', 'sunshine','num_day','energy_num', 'temp_num', 'humid_num', 'sunshine_num', 'temp4',
       'energy_num_weekday', 'energy_day_lag_1', 'energy_day_lag_2',
       'energy_day_lag_3', 'energy_day_lag_4', 'energy_day_lag_5',
       'energy_day_lag_6', 'energy_day_lag_7', 'energy_day_lag_14',
       'energy_day_lag_21']]

# THI 포함
X_train_cat = X_train[['num',
       'day', 'hour', 'weekday',  'isholy','group']]
# X_train_cat = X_train[['num','cooling', 'solar',
#        'month', 'day', 'hour', 'weekday',  'isholy', 'isweekend',
#         'group']]
X_train = X_train[['temp', 'wind', 'humid', 'rain', 'sunshine','num_day','energy_num', 'temp_num', 'humid_num', 'sunshine_num', 'temp4',
       'energy_num_weekday', 'energy_day_lag_1', 'energy_day_lag_2',
       'energy_day_lag_3', 'energy_day_lag_4', 'energy_day_lag_5',
       'energy_day_lag_6', 'energy_day_lag_7', 'energy_day_lag_14',
       'energy_day_lag_21','THI']]

X_valid_cat = X_valid[['num',
       'day', 'hour', 'weekday',  'isholy','group']]
# X_valid_cat = X_valid[['num','cooling', 'solar',
#        'month', 'day', 'hour', 'weekday',  'isholy', 'isweekend',
#         'group']]
X_valid = X_valid[['temp', 'wind', 'humid', 'rain', 'sunshine','num_day','energy_num', 'temp_num', 'humid_num', 'sunshine_num', 'temp4',
       'energy_num_weekday', 'energy_day_lag_1', 'energy_day_lag_2',
       'energy_day_lag_3', 'energy_day_lag_4', 'energy_day_lag_5',
       'energy_day_lag_6', 'energy_day_lag_7', 'energy_day_lag_14',
       'energy_day_lag_21','THI']]

X_test_cat = X_test[['num',
       'day', 'hour', 'weekday',  'isholy','group']]
# X_test_cat = X_test[['num','cooling', 'solar',
#        'month', 'day', 'hour', 'weekday',  'isholy', 'isweekend',
#         'group']]
X_test = X_test[['temp', 'wind', 'humid', 'rain', 'sunshine','num_day','energy_num', 'temp_num', 'humid_num', 'sunshine_num', 'temp4',
       'energy_num_weekday', 'energy_day_lag_1', 'energy_day_lag_2',
       'energy_day_lag_3', 'energy_day_lag_4', 'energy_day_lag_5',
       'energy_day_lag_6', 'energy_day_lag_7', 'energy_day_lag_14',
       'energy_day_lag_21','THI']]

from sklearn.preprocessing import MaxAbsScaler

ma_scaler = MaxAbsScaler()
X_train_ma = ma_scaler.fit_transform(X_train)
X_valid_ma = ma_scaler.fit_transform(X_valid)
X_test_ma = ma_scaler.fit_transform(X_test)

# 스케일링 후 붙이기
X_train_cat = X_train_cat.reset_index(drop=True)
X_train_ma = pd.DataFrame(X_train_ma)
X_train_ma = pd.concat([X_train_ma, X_train_cat], axis = 1)
X_train_ma
X_train_cat
X_valid_cat = X_valid_cat.reset_index(drop=True)
X_valid_ma = pd.DataFrame(X_valid_ma)
X_valid_ma = pd.concat([X_valid_ma, X_valid_cat], axis = 1)
X_valid_ma
X_test_cat = X_test_cat.reset_index(drop=True)
X_test_ma = pd.DataFrame(X_test_ma)
X_test_ma = pd.concat([X_test_ma, X_test_cat], axis = 1)
X_test_ma
X_test_cat
X_train_cat
X_train_ma.shape
X_test_ma
X_test_cat = X_test_cat.reset_index(drop=True)


####

# from sklearn.preprocessing import RobustScaler

# rb_scaler = RobustScaler()
# X_train_rb = rb_scaler.fit_transform(X_train)
# X_valid_rb = rb_scaler.fit_transform(X_valid)


#########################

from sklearn.decomposition import PCA
pca = PCA(n_components = 10)
pca.fit(X_train)
print(pca.components_.shape)
X_train_pca = pca.transform(X_train)
print(X_train_pca)
print(X_train_pca.shape)
''''''
# 그룹 만들기
from sklearn.preprocessing import MaxAbsScaler
def cluster_df(scaler=MaxAbsScaler()): # scaler=[False,'MinMaxScaler()','StandardScaler()']
    train_=train.copy()
    train_ts=train_.pivot_table(values='energy', index=train_.num,columns='date_time',aggfunc='first')

    if scaler:
        train_ts_T=scaler.fit_transform(train_ts.T)
        train_ts=pd.DataFrame(train_ts_T.T,index=train_ts.index,columns=train_ts.columns)

    return train_ts


def visualize_n_cluster(train_ts, n_lists=[3,4,5,6],metric='dtw',seed=2021,vis=True):

    for idx,n in enumerate(n_lists):
        ts_kmeans=TimeSeriesKMeans(n_clusters=n, metric=metric, random_state=seed)
        train_ts['cluster(n={})'.format(n)]=ts_kmeans.fit_predict(train_ts)
        score=round(silhouette_score(train_ts,train_ts['cluster(n={})'.format(n)],metric='euclidean'),3)

    return train_ts

train_ts = cluster_df()
train_ts = visualize_n_cluster(train_ts, n_lists = [3,4,5,6], metric='euclidean', seed = 2021, vis = True)
train_group = train_ts['cluster(n=4)']
train_group

train = pd.merge(train, train_group, on='num', how = 'left')
# THI 포함
train.columns = ['num', 'date_time', 'energy', 'temp', 'wind', 'humid', 'rain',
       'sunshine', 'cooling', 'solar', 'month', 'day', 'hour', 'weekday',
       'num_day', 'isholy', 'isweekend', 'energy_num', 'temp_num', 'humid_num',
       'sunshine_num', 'temp4', 'energy_num_weekday', 'THI',
       'energy_day_lag_1', 'energy_day_lag_2', 'energy_day_lag_3',
       'energy_day_lag_4', 'energy_day_lag_5', 'energy_day_lag_6',
       'energy_day_lag_7', 'energy_day_lag_14', 'energy_day_lag_21',
       'group']

# THI 제외
train.columns = ['num', 'date_time', 'energy', 'temp', 'wind', 'humid', 'rain',
       'sunshine', 'cooling', 'solar', 'month', 'day', 'hour', 'weekday',
       'num_day', 'isholy', 'isweekend', 'energy_num', 'temp_num', 'humid_num',
       'sunshine_num', 'temp4', 'energy_num_weekday', 'energy_day_lag_1',
       'energy_day_lag_2', 'energy_day_lag_3', 'energy_day_lag_4',
       'energy_day_lag_5', 'energy_day_lag_6', 'energy_day_lag_7',
       'energy_day_lag_14', 'energy_day_lag_21', 'group']
train

''''''


#########################
#########################
#########################

from xgboost import XGBRegressor
xgb = XGBRegressor(
    max_depth = 10,
    n_estimators = 1000,
    min_child_weight=300,
    colsample_bytree = 0.8,
    subsample=0.8,
    eta=0.3,
    seed=42)

xgb.fit(
    X_train_ma,
    y_train,
    # eval_metric='rmse',
    # eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=1,
    # early_stopping_rounds=30
)

y_pred = xgb.predict(X_valid_ma)
y_test_xgboost = xgb.predict(X_test_ma)

##################################
from xgboost import plot_importance
fig, ax = plt.subplots(1, 1, figsize = (10, 14))
plot_importance(xgb, ax = ax)
X_train
# 모델 저장하기
# model.save_model('.model')

# 제출자료 만들기
submission = pd.DataFrame({
    "num_date_time": sample_submission.num_date_time, 
    "answer": y_test_xgboost
})
submission.to_csv('xgb_submission_minabs_scaler_grouping_cat_cut.csv', index=False)
sample_submission
submission

'''
1회차.
xgb_submission.csv
피클 불러오자마자 난 그대로 두고 xg부스트 돌려보기. 생각보다 점수 잘 나옴. 
점수 : 22.7423089706
등수 : 제출 시 219등

2회차.
xgb_submission_2.csv
모델은 얼리스탑 30->50, 불쾌지수와 hour lag 적용해서 다시 돌려봄.
점수 : 73.9680038861	

3회차.
lgbm_submission.csv
lgbm.. 기타 기록은 누락.
점수 : 52.0971434916	

4회차.
rf_submission.csv
랜덤 포레스트..
점수 : 38.3784020047	

4회차.
xgb_submission_3.csv
모델은 얼리스탑 30, 불쾌지수와 hour lag 빼고, nan값 0으로 채움.
점수 : ??? 이건 뭐지.. 기록 누락..

5회차.
xgb_submission_4.csv
모델은 4와 그대로.. temp4만 올려서 돌렸는데, 연관성 바닥을 치네. 버리자.


6회차.
gbm_submission.csv
gbm 모델 가져다 썼고, 성능은 그냥 그렇네.. 아무래도 이후 추가한 변수를 버려야겠다.
점수 : 44.2574092616

7회차.
앙상블
voting_ensemble_submission.csv
점수 : 92.4047396438
왜 점점 퇴화하냐..

10회차
xgboost
3시간 예측, 불쾌지수 빼니까 등수가 올라감... 
+ valid 통째로 빼버림.
점수 : 19.6489473422

그래.. 당일거를 예측에 쓰는건 확실히 뭔가 안맞는 느낌이 있지.. 근데 그런의미에서 하루 전도 틀린거 아닐까 싶네
템프4를 살려볼까?

11회차
xgboost
3시간 예측, 불쾌지수 빼니까 등수가 올라감... 
+ valid 통째로 빼버림.
+ MaxAbsScaler 적용
점수 : 16.8679736883

스케일러 여러개 돌려본 보람이 있네.

12회차
xgboost
3시간 예측, 불쾌지수 빼니까 등수가 올라감... 
+ valid 통째로 빼버림.
+ MaxAbsScaler 적용
+ 그룹화 결과도 붙일 예정
점수 : 16.0963495967	

아주 미약하게 상승하는 효과가 있음... 쓸데없는걸 빼볼까 고민.

13회차
xgboost
3시간 예측, 불쾌지수 빼니까 등수가 올라감... 
+ valid 통째로 빼버림.
+ MaxAbsScaler 적용
+ 그룹화 결과도 붙일 예정
+ 카테고리 빼고 스케일링 했는데 떨어짐
점수 : 17.6096165915

뭐여 이건.. 14회차는 쓸데없는 카테고리 싹 빼버리고, k폴드 돌려봐야겠다

'''
###############################
###############################
###############################
###############################

#########################
# ---*---*---*  ---*---*---*
# 세번째 제출 준비
# 불쾌지수, hour lag 추가, nan값 제거, lgbm으로

from lightgbm import LGBMRegressor
lgbm = LGBMRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3,
    seed=42
)
lgbm.fit(
    X_train_ma,
    y_train,
    #eval_metric='rmse',
    #eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=True,
    #early_stopping_rounds=30
)

y_pred = lgbm.predict(X_valid_ma)
y_test_lgbm = lgbm.predict(X_test_ma)
y_test_lgbm

# 제출자료 만들기
submission = pd.DataFrame({
    "num_date_time": sample_submission.num_date_time, 
    "answer": y_test_lgbm
})
submission.to_csv('lgbm_submission_3.csv', index=False)
sample_submission
submission

# 중요도 확인
from lightgbm import plot_importance
fig, ax = plt.subplots(1, 1, figsize = (10, 14))
plot_importance(lgbm, ax = ax)

'''
점수 : 52.0971434916	
이건 뭐지... 왜 더 떨어질까.
'''


# ---*---*---*  ---*---*---*
# 네번째 제출 준비
# 불쾌지수, hour lag 추가, 스케일러는 미적용
# 단순 랜덤포레스트 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(
    n_estimators=25,
    random_state=42, 
    max_depth=15, 
    n_jobs=-1
)
rf.fit(
    X_train,
    y_train,
)

y_pred = rf.predict(X_valid)
y_test_rf = rf.predict(X_test)
y_test_rf

# 제출자료 만들기
submission = pd.DataFrame({
    "num_date_time": sample_submission.num_date_time, 
    "answer": y_test_rf
})
submission.to_csv('rf_submission.csv', index=False)
sample_submission
submission

# ---*---*---*  ---*---*---*
# 여섯번째 제출 준비
import lightgbm as lgb
feature_name = X_train.columns.tolist()
X_train.num_day
params = {
    'objective': 'mse',
    'metric': 'rmse',
    'num_leaves': 2 ** 8 -1,
    'learning_rate': 0.005,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.75,
    'bagging_freq': 5,
    'seed': 1,
    'verbose': 1
}
feature_name_indexes = [ 
                        'isholy',
                        'weekday',
                        'isweekend',
]

lgb_train = lgb.Dataset(X_train[feature_name], y_train)
lgb_eval = lgb.Dataset(X_valid[feature_name], y_valid, reference=lgb_train)

evals_result = {}
gbm = lgb.train(
        params, 
        lgb_train,
        num_boost_round=3000,
        valid_sets=(lgb_train, lgb_eval), 
        feature_name = feature_name,
        categorical_feature = feature_name_indexes,
        verbose_eval=50, 
        evals_result = evals_result,
        early_stopping_rounds = 30)

y_test = gbm.predict(X_test[feature_name])


# 제출자료 만들기
submission = pd.DataFrame({
    "num_date_time": sample_submission.num_date_time, 
    "answer": y_test
})
submission.to_csv('gbm_submission.csv', index=False)



# ---*---*---*  ---*---*---*
# 일곱번째 제출 준비

#앙상블 모델

# 함수
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

my_predictions = {}
my_pred = None
my_actual = None
my_name = None

colors = ['r', 'c', 'm', 'y', 'k', 'khaki', 'teal', 'orchid', 'sandybrown',
          'greenyellow', 'dodgerblue', 'deepskyblue', 'rosybrown', 'firebrick',
          'deeppink', 'crimson', 'salmon', 'darkred', 'olivedrab', 'olive', 
          'forestgreen', 'royalblue', 'indigo', 'navy', 'mediumpurple', 'chocolate',
          'gold', 'darkorange', 'seagreen', 'turquoise', 'steelblue', 'slategray', 
          'peru', 'midnightblue', 'slateblue', 'dimgray', 'cadetblue', 'tomato'
         ]

def plot_predictions(name_, pred, actual):
    df = pd.DataFrame({'prediction': pred, 'actual': y_valid})
    df = df.sort_values(by='actual').reset_index(drop=True)

    plt.figure(figsize=(11, 8))
    plt.scatter(df.index, df['prediction'], marker='x', color='r')
    plt.scatter(df.index, df['actual'], alpha=0.7, marker='o', color='black')
    plt.title(name_, fontsize=15)
    plt.legend(['prediction', 'actual'], fontsize=12)
    plt.show()

def mse_eval(name_, pred, actual):
    global my_predictions, colors, my_pred, my_actual, my_name
    
    my_name = name_
    my_pred = pred
    my_actual = actual

    plot_predictions(name_, pred, actual)

    mse = mean_squared_error(pred, actual)
    my_predictions[name_] = mse

    y_value = sorted(my_predictions.items(), key=lambda x: x[1], reverse=True)
    
    df = pd.DataFrame(y_value, columns=['model', 'mse'])
    print(df)
    min_ = df['mse'].min() - 10
    max_ = df['mse'].max() + 10
    
    length = len(df) / 2
    
    plt.figure(figsize=(9, length))
    ax = plt.subplot()
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['model'], fontsize=12)
    bars = ax.barh(np.arange(len(df)), df['mse'], height=0.3)
    
    for i, v in enumerate(df['mse']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        ax.text(v + 2, i, str(round(v, 3)), color='k', fontsize=12, fontweight='bold', verticalalignment='center')
        
    plt.title('MSE Error', fontsize=16)
    plt.xlim(min_, max_)
    
    plt.show()
    
def add_model(name_, pred, actual):
    global my_predictions, my_pred, my_actual, my_name
    my_name = name_
    my_pred = pred
    my_actual = actual
    
    mse = mean_squared_error(pred, actual)
    my_predictions[name_] = mse

def remove_model(name_):
    global my_predictions
    try:
        del my_predictions[name_]
    except KeyError:
        return False
    return True

def plot_all():
    global my_predictions, my_pred, my_actual, my_name
    
    plot_predictions(my_name, my_pred, my_actual)
    
    y_value = sorted(my_predictions.items(), key=lambda x: x[1], reverse=True)
    
    df = pd.DataFrame(y_value, columns=['model', 'mse'])
    print(df)
    min_ = df['mse'].min() - 10
    max_ = df['mse'].max() + 10
    
    length = len(df) / 2
    
    plt.figure(figsize=(9, length))
    ax = plt.subplot()
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['model'], fontsize=12)
    bars = ax.barh(np.arange(len(df)), df['mse'], height=0.3)
    
    for i, v in enumerate(df['mse']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        ax.text(v + 2, i, str(round(v, 3)), color='k', fontsize=12, fontweight='bold', verticalalignment='center')
        
    plt.title('MSE Error', fontsize=16)
    plt.xlim(min_, max_)
    
    plt.show()


# 모델
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image

# 회귀
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
pred = linear_reg.predict(X_valid)
mse_eval('LinearRegression', pred, y_valid)

#릿지
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
pred = ridge.predict(X_valid)
mse_eval('Ridge(alpha=0.1)', pred, y_valid)

#라쏘
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
pred = lasso.predict(X_valid)
mse_eval('Lasso(alpha=0.01)', pred, y_valid)

#엘라스틱
elasticnet = ElasticNet(alpha=0.01, l1_ratio=0.8)
elasticnet.fit(X_train, y_train)
pred = elasticnet.predict(X_valid)
mse_eval('ElasticNet(alpha=0.1, l1_ratio=0.8)', pred, y_valid)    

#파이프라인
elasticnet_pipeline = make_pipeline(
    StandardScaler(),
    ElasticNet(alpha=0.01, l1_ratio=0.8)
)
elasticnet_pipeline.fit(X_train, y_train)
elasticnet_pred = elasticnet_pipeline.predict(X_valid)
mse_eval('Standard ElasticNet', elasticnet_pred, y_valid)

#폴리 파이프라인
poly_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    ElasticNet(alpha=0.1, l1_ratio=0.2)
)
poly_pipeline.fit(X_train, y_train)
poly_pred = poly_pipeline.predict(X_valid)
mse_eval('Poly ElasticNet', poly_pred, y_valid)

single_models = [
    ('linear_reg', linear_reg), 
    ('ridge', ridge), 
    ('lasso', lasso), 
    ('elasticnet_pipeline', elasticnet_pipeline), 
    ('poly_pipeline', poly_pipeline)
]
voting_regressor = VotingRegressor(single_models, n_jobs=-1)

VotingRegressor(estimators=[('linear_reg',
                             LinearRegression(copy_X=True, fit_intercept=True,
                                              n_jobs=None, normalize=False)),
                            ('ridge',
                             Ridge(alpha=0.1, copy_X=True, fit_intercept=True,
                                   max_iter=None, normalize=False,
                                   random_state=None, solver='auto',
                                   tol=0.001)),
                            ('lasso',
                             Lasso(alpha=0.01, copy_X=True, fit_intercept=True,
                                   max_iter=1000, normalize=False,
                                   positive=False,
                                      steps=[('polynomialfeatures',
                                              PolynomialFeatures(degree=2,
                                                                 include_bias=False,
                                                                 interaction_only=False,
                                                                 order='C')),
                                             ('elasticnet',
                                              ElasticNet(alpha=0.1, copy_X=True,
                                                         fit_intercept=True,
                                                         l1_ratio=0.2,
                                                         max_iter=1000,
                                                         normalize=False,
                                                         positive=False,
                                                         precompute=False,
                                                         random_state=None,
                                                         selection='cyclic',
                                                         tol=0.0001,
                                                         warm_start=False))],
                                      verbose=False))],
                n_jobs=-1, weights=None)


voting_regressor.fit(X_train, y_train)

voting_pred = voting_regressor.predict(X_test)
voting_pred = voting_regressor.predict(X_valid)

mse_eval('Voting Ensemble', voting_pred, y_valid)


# 제출자료 만들기
submission = pd.DataFrame({
    "num_date_time": sample_submission.num_date_time, 
    "answer": voting_pred
})
submission.to_csv('voting_ensemble_submission.csv', index=False)

# ---*---*---*  ---*---*---*
# 여덟번째 제출 준비
# valid 없이, 노멀 lgbm 제출
from lightgbm import LGBMRegressor

# LightGBM Regressor 모델
lgbm = LGBMRegressor(max_depth=10, n_estimators=100, random_state=40)
lgbm.fit(X_train, y_train)

# 예측하기
y_pred = lgbm.predict(X_test)

# 제출자료 만들기
submission = pd.DataFrame({
    "num_date_time": sample_submission.num_date_time, 
    "answer": y_pred
})
submission.to_csv('lgbm_submission_2.csv', index=False)

# ---*---*---*  ---*---*---*
# 아홉번째 제출 준비
# xgb

from xgboost import XGBRegressor
xgb = XGBRegressor(
    max_depth = 10,
    n_estimators = 1000,
    seed=42)

xgb.fit(
    X_train,
    y_train,
    verbose=True,
)

y_test_xgboost = xgb.predict(X_test)

# 제출자료 만들기
submission = pd.DataFrame({
    "num_date_time": sample_submission.num_date_time, 
    "answer": y_test_xgboost
})
submission.to_csv('xgb_submission_5.csv', index=False)


# ---*---*---*  ---*---*---*
# 여덟번째 제출 준비
# kfold로 도전
from sklearn.model_selection import KFold


train_x = train[train.num_day < 86].drop(['energy'], axis=1)
train_y = train[train.num_day < 86]['energy']

X_test = train[train.num_day >= 86].drop(['energy'], axis=1)


cross=KFold(n_splits=5, shuffle=True, random_state=42)
folds=[]
for train_idx, valid_idx in cross.split(train_x, train_y):
    folds.append((train_idx, valid_idx))

models={}
for fold in range(5):
    print(f'===================={fold+1}=======================')
    train_idx, valid_idx=folds[fold]
    X_train=train_x.iloc[train_idx, :]
    y_train=train_y.iloc[train_idx, :]
    X_valid=train_x.iloc[valid_idx, :]
    y_valid=train_y.iloc[valid_idx, :]
    
    model=LGBMRegressor(n_estimators=100)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], 
             early_stopping_rounds=30, verbose=100)
    models[fold]=model
    
    print(f'================================================\n\n')

for i in range(5):
    submission['answer'] += models[i].predict(test)/5 




# ---*---*---*  ---*---*---*
# N번째 제출 준비
# 불쾌지수, hour lag 추가, 스케일러 적용
from sklearn.preprocessing import MinMaxScaler

mmscaler = MinMaxScaler()

train_mmscaled = mmscaler.fit_transform(train)

train_mmscaled_undo = mmscaler.inverse_transform(train_mmscaled)
train_mmscaled_undo

























# ---*---*---*  ---*---*---*
# ---*---*---*  ---*---*---*
# ---*---*---*  ---*---*---*
'''
전체 틀잡기
'''
from xgboost import XGBRegressor
model = XGBRegressor(
    max_depth = 8,
    n_estimators = 1000,
    min_child_weight=300,
    colsample_bytree = 0.8,
    subsample=0.8,
    eta=0.3,
    seed=42)

model.fit(
    X_train,
    y_train,
    eval_metric='rmse',
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=True,
    early_stopping_rounds=30
)

y_pred = model.predict(X_valid)
y_test_xgboost = model.predict(X_test)

##################################

from lightgbm import LGBMRegressor
lgbm = LGBMRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3,
    seed=42
)
lgbm.fit(
    X_train,
    y_train,
    eval_metric='rmse',
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=True,
    early_stopping_rounds=30
)

y_test_lgbm = lgbm.predict(X_test)
y_test_lgbm

#####################################

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(
    n_estimators=25,
    random_state=42, 
    max_depth=15, 
    n_jobs=-1
)
rf.fit(
    X_train,
    y_train,
)

y_pred = rf.predict(X_valid)
y_test_rf = rf.predict(X_test)
y_test_rf
#################################


from xgboost import plot_importance
fig, ax = plt.subplots(1, 1, figsize = (10, 14))
plot_importance(model, ax = ax)



from lightgbm import plot_importance
fig, ax = plt.subplots(1, 1, figsize = (10, 14))
plot_importance(lgbm, ax = ax)



# 모델 저장하기
# model.save_model('.model')



# 제출자료 만들기
submission = pd.DataFrame({
    "num_date_time": sample_submission.num_date_time, 
    "answer": y_test_xgboost
})
submission.to_csv('xgb_submission.csv', index=False)
sample_submission
submission











################
