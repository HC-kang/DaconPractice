import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import date
import matplotlib.pyplot as plt
from matplotlib import rc
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
# from glob import glob
import plotly.express as px
import os
import seaborn as sns

os.chdir('/Users/heechankang/projects/pythonworkspace/dacon_data/energy_consume')
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

train = pd.read_csv('train.csv', encoding='cp949')
test = pd.read_csv('test.csv', encoding='cp949')
sample_submission = pd.read_csv('sample_submission.csv', encoding='cp949')

train.head() 
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

data = train[['date_time', 'energy']]

# 월별 나누기
data['month'] = data['date_time'].dt.month
data
# 월별 그래프
mean_month = data.groupby('month').mean()
fig=px.bar(mean_month, x=mean_month.index, y='energy')
fig.show()

# 요일별 나누기
data['weekday'] = data['date_time'].dt.weekday
data
# 요일별 그래프
mean_weekday = data.groupby('weekday').mean()
fig=px.bar(mean_weekday, x=mean_weekday.index, y='energy')
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

# 1. temp
nan_feature(test, 'temp')

# 2. wind
nan_feature(test, 'wind')

# 3. humid
nan_feature(test, 'humid')

# 4. rain
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

for i in range(60):
    print(test.cooling[i*168]-test.cooling[i*168+165])

test.cooling[0:5] = 0
test.loc[0:5, 'cooling'] = 0

test[test['num']==60]
for i in range(60):
    if test.loc[i*168, 'cooling']==1:
        test.loc[i*168:i*168+167, 'cooling']=1
test.cooling.sum()
test.solar.sum()
len(test.num.unique())

test
for i in range(60):
    print(test[test['num']==i]['cooling'].sum())

# 계산 및 전처리를 위해 임시적으로 test를 train에 concat
train = pd.concat(train, test)



# 시간별 나누기
data['hour'] = data['date_time'].dt.hour
data
sns.lineplot(data = data)
# 시간대별 그래프
mean_hour = data.groupby('hour').mean()
fig=px.bar(mean_hour, x=mean_hour.index, y='energy')
fig.show()

# date_time 변수 쪼개기
train['month'] = train['date_time'].dt.month
train['day'] = train['date_time'].dt.day
train['hour'] = train['date_time'].dt.hour
train['num_day'] = train['date_time'].dt.dayofyear - 152
train

# 새로운 변수 생성 - 시간과 건물에 따른 평균치
group = train.groupby(['date_time', 'num']).mean()
group
group.drop(['wind', 'rain', 'cooling', 'solar'])

