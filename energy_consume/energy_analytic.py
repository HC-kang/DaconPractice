import pandas as pd
import numpy as np
import math
import os

#Visualizing
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns; #sns.set_style('whitegrid')

#Time Series Analysis
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, ccf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#Clustering (+α)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from minisom import MiniSom

#Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout

#System
from ipywidgets import interact
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

# Custom Loss Function
import tensorflow.keras.backend as K

#TODO: 안그래도 만들어야겠다고 생각했던 공식인데, 가져가자
def SMAPE(true,predicted):
    epsilon = 0.1
    summ = K.maximum(K.abs(true) + K.abs(predicted) + epsilon, 0.5 + epsilon)
    smape = K.abs(predicted - true) / summ * 2.0
    return smape


os.chdir('/Users/heechankang/projects/pythonworkspace/dacon_data/energy_consume')

train = pd.read_csv('train.csv', encoding='cp949',parse_dates=['date_time'])
test = pd.read_csv('test.csv', encoding='cp949',parse_dates=['date_time'])
submission = pd.read_csv('sample_submission.csv', encoding='cp949')


# 요일
def weekday(x):
    if x.dayofweek == 6:
        return '일'
    elif x.dayofweek == 0:
        return '월'
    elif x.dayofweek == 1:
        return '화'
    elif x.dayofweek == 2:
        return '수'
    elif x.dayofweek == 3:
        return '목'
    elif x.dayofweek == 4:
        return '금'
    else:
        return '토'

# 주말 여부
def weekend(x):
    if x.dayofweek in [5,6]:
        return 1
    else:
        return 0

train['month'] = train['date_time'].dt.month
train['day'] = train['date_time'].dt.day
train['hour'] = train['date_time'].dt.hour
train['weekday'] = train['date_time'].apply(weekday)
train['weekend'] = train['date_time'].apply(weekend)

test['month'] = test['date_time'].dt.month
test['day'] = test['date_time'].dt.day
test['hour'] = test['date_time'].dt.hour
test['weekday'] = test['date_time'].apply(weekday)
test['weekend'] = test['date_time'].apply(weekend)


print('\033[1m<test함수 변수별 결측값 수>\033[0m\n', test.isna().sum())

# 결측치 채우기
building_info = train[['num', '비전기냉방설비운영', '태양광보유']].drop_duplicates()

test.drop(columns=['비전기냉방설비운영', '태양광보유'], inplace = True)

test = pd.merge(test, building_info, on='num')
test

train['일조(hr)'].value_counts()

test.tail(6)
test['일조(hr, 3시간)'] = test['일조(hr, 3시간)'].interpolate(method = 'pad')

train_ = train.copy()

def make_train_nan(col, n):
    new_list = []
    for idx, temp in enumerate(train_[col]):
        if idx%n == 0:
            new_list.append(temp)
        else:
            new_list.append(np.nan)
    train_['{}'.format(col+'_nan')] = new_list

make_train_nan('기온(°C)',3)
make_train_nan('풍속(m/s)',3)
make_train_nan('습도(%)',3)
make_train_nan('강수량(mm)',6)

print(train_.iloc[:, -4:].isnull().sum())


def compare_interpolate_methods(col, methods, metric):
    error_dict = dict()
    for method in methods:
        fillna = train_['{}'.format(col+'_nan')].interpolate(method = method)
        if fillna.isna().sum()!=0:
            fillna = fillna.interpolate(method = 'linear')
        error = metric(train_['{}'.format(col)], fillna)
        error_dict['{}'.format(method)] = error

    return error_dict
#TODO: 재미있어보여서 반영하고싶음.
all_cols_error_dict = dict()
for col in ['기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)']:
    methods = ['pad', 'linear', 'quadratic', 'cubic', 'values']
    error_dict = compare_interpolate_methods(col, methods, mean_squared_error)
    all_cols_error_dict['{}'.format(col)] = error_dict

all_cols_error_df = pd.DataFrame(all_cols_error_dict)

fig, axes = plt.subplots(1,4, figsize = (18,5), sharey=False)
for i in range(len(all_cols_error_df.columns)):
    sns.lineplot(ax=axes[i], data = all_cols_error_df.iloc[:,i].transpose())

# 기온 결측치
test['기온(°C)']=test['기온(°C)'].interpolate(method='quadratic')
# 마지막 na 채우기
test['기온(°C)']=test['기온(°C)'].interpolate(method='linear')
test

# 풍속 결측치 채우기
test['풍속(m/s)']=test['풍속(m/s)'].interpolate(method='linear')

#습도 결측치 채우기
test['습도(%)'] = test['습도(%)'].interpolate(method='quadratic')
#마지막 na 채우기
test['습도(%)'] = test['습도(%)'].interpolate(method='linear')

#강수량 결측치 채우기
test['강수량(mm, 6시간)'] = test['강수량(mm, 6시간)'].interpolate(method='linear')

train['불쾌지수'] = 1.8*train['기온(°C)'] - 0.55*(1-(train['습도(%)']/100))*(1.8*train['기온(°C)']-26) + 32
test['불쾌지수'] = 1.8*test['기온(°C)'] - 0.55*(1-(test['습도(%)']/100))*(1.8*test['기온(°C)']-26) + 32

# (데이터프레임 변수 순서 정리)
train = train[['num','date_time','month', 'day', 'hour','weekday','weekend','기온(°C)','습도(%)','불쾌지수','풍속(m/s)','강수량(mm)','일조(hr)','비전기냉방설비운영','태양광보유','전력사용량(kWh)']]
test = test[['num','date_time','month', 'day', 'hour','weekday','weekend','기온(°C)','습도(%)','불쾌지수','풍속(m/s)','강수량(mm, 6시간)','일조(hr, 3시간)','비전기냉방설비운영','태양광보유']]



#####
# 전력사용량 EDA
###

''' 건물들의 평균 전력사용량 '''
sns.histplot(train.groupby('num')['전력사용량(kWh)'].mean())

train.groupby(['비전기냉방설비운영', '태양광보유'])['전력사용량(kWh)'].mean()


corr = []
for num in range(1, 61):
    df = train[train.num==num]
    num_corr = df.corr()['전력사용량(kWh)']
    num_corr = num_corr.drop(['num', '비전기냉방설비운영', '태양광보유', '전력사용량(kWh)'])
    corr.append(num_corr)
corr_df = pd.concat(corr, axis = 1).T
corr_df.index = list(range(1, 61))

# 시각화
f, ax = plt.subplots(figsize = (20, 8))
plt.title('건물별 전력사용량과 변수들의 상관관게', fontsize = 15)
sns.heatmap(corr_df.T, cmap = sns.diverging_palette(240, 10, as_cmap=True), ax = ax)
plt.xlabel('건물(num)')
plt.show()

def vis_time_series_decompose(num):
    df = train[train.num==num]
    df.index=df.date_time

    res=sm.tsa.seasonal_decompose(df['전력사용량(kWh)'], model='additive')

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize = (20, 12))
    res.observed.plot(ax=ax1, title='Observed')
    res.trend.plot(ax=ax2, title='Trend')
    res.resid.plot(ax=ax3, title='Residual')
    res.seasonal.plot(ax=ax4, title='Seasonal')
    plt.tight_layout()
    plt.show()

vis_time_series_decompose(num=4)


vis_time_series_decompose(num=9)


df = train[train.num==4]
fig, (ax1, ax2)=plt.subplots(2,1,figsize=(10, 6))
plot_acf(df['전력사용량(kWh)'], lags = 50, ax = ax1)
plot_pacf(df['전력사용량(kWh)'], lags=50, ax = ax2)
plt.tight_layout()
plt.show()



df = train[train.num==9]
fig, (ax1, ax2)=plt.subplots(2,1,figsize=(10, 6))
plot_acf(df['전력사용량(kWh)'], lags = 50, ax = ax1)
plot_pacf(df['전력사용량(kWh)'], lags=50, ax = ax2)
plt.tight_layout()
plt.show()



df = train[train.num==58]
fig, (ax1, ax2)=plt.subplots(2,1,figsize=(10, 6))
plot_acf(df['전력사용량(kWh)'], lags = 50, ax = ax1)
plot_pacf(df['전력사용량(kWh)'], lags=50, ax = ax2)
plt.tight_layout()
plt.show()


# 시간별 평균 전력사용량 시각화
fig = plt.figure(figsize = (15, 15))
plt.title('각 건물의 시간별 평균 전력사용량(kWh)', fontsize=15, y=1.05)
plt.axis('off')

for num in range(1, 61):
    df = train[train.num==num]
    ax = fig.add_subplot(10, 6, num)
    ax.plot(df['hour'].unique(), df.groupby('hour')['전력사용량(kWh)'].mean())
    ax.set_title(f'건물:{num}')
    ax.set_xticks([0, 6, 12, 18,24])
plt.tight_layout()
plt.show()


fig = plt.figure(figsize = (15,15))
plt.title('각 건물의 요일별 평균 전력 사용량(kWh)', fontsize=15,y=1.05)
plt.axis('off')

weekday=['월','화','수','목','금','토','일']
colors=['skyblue','skyblue','skyblue','skyblue','skyblue','pink','pink']

for num in range(1,61):
    df=train[train.num==num]
    df_counts=df.groupby('weekday')['전력사용량(kWh)'].mean()
    df_counts=df_counts.reindex(weekday)
    ax=fig.add_subplot(10,6,num)
    ax.bar(df['weekday'].unique(),df_counts,color=colors)
    ax.set_title(f'건물:{num}')
plt.tight_layout()
plt.show()

# 시간, 요일별 평균 전력사용량
fig=plt.figure(figsize=(15,15))
plt.title('각 건물의 시간에 따른 주말여부 평균 전력 사용량(kWh)',fontsize=15,y=1.05)
plt.axis('off')

for num in range(1,61):
    df=train[train.num==num]
    ax=fig.add_subplot(10,6,num)
    ax.plot(df['hour'].unique(),df[df.weekend==0].groupby('hour')['전력사용량(kWh)'].mean(),label='평일')
    ax.plot(df['hour'].unique(),df[df.weekend==1].groupby('hour')['전력사용량(kWh)'].mean(),label='주말')
    ax.set_title(f'건물:{num}')
    ax.set_xticks([0,6,12,18,24])
lines,labels=fig.axes[-1].get_legend_handles_labels()
fig.legend(lines,labels,loc=1,prop={'size':12})
plt.tight_layout()
plt.show()


#####
# 군집화
###

## 시계열 군집화용 DataFrame을 생성하는 함수(row->num, col->date_time)
def cluster_df(scaler=MinMaxScaler()): # scaler=[False,'MinMaxScaler()','StandardScaler()']
    train_=train.copy()
    train_ts=train_.pivot_table(values='전력사용량(kWh)', index=train_.num,columns='date_time',aggfunc='first')

    if scaler:
        train_ts_T=scaler.fit_transform(train_ts.T)
        train_ts=pd.DataFrame(train_ts_T.T,index=train_ts.index,columns=train_ts.columns)

    return train_ts

## SOM 알고리즘 결과를 정리해주는 DataFrame을 생성하는 함수
def make_som_df(X):
    win_map=som.win_map(X)
    som_result=[]
    for i in range(60):
        som_result.append([i+1,som.winner(X[i])])
    som_df=pd.DataFrame(som_result,columns=['num','cluster'])
    return som_df

## 여러 n_cluster에 대해 TimeSeriesKmeans를 시행하고 이를 시각화하는 함수
def visualize_n_cluster(train_ts, n_lists=[3,4,5,6],metric='dtw',seed=2021,vis=True):

    if vis:
        fig=plt.figure(figsize=(20,5))
        plt.title('군집 개수별 건물수 분포',fontsize=15,y=1.2)
        plt.axis('off')

    for idx,n in enumerate(n_lists):
        ts_kmeans=TimeSeriesKMeans(n_clusters=n, metric=metric, random_state=seed)
        train_ts['cluster(n={})'.format(n)]=ts_kmeans.fit_predict(train_ts)
        score=round(silhouette_score(train_ts,train_ts['cluster(n={}'.format(n)],metric='euclidean'),3)

        vc=train_ts['cluster(n={})'.format(n)].value_counts()

        if vis:
            ax=fig.add_subplot(1,len(n_lists),idx+1)
            sns.barplot(x=vc.index,y=vc,palette='Pastel1')
            ax.set(title='n_cluster={0}\nscore:{1}'.format(n, score))
    if vis:
        plt.tight_layout()
        plt.show()

    return train_ts


## 클러스터 별 시간&요일별 전력사용량 평균 시각화(new)
def visualize_by_cluster(df_with_labels, n_cluster, algorithm):
    fig=plt.figure(figsize=(20, 4*n_cluster))
    plt.title('군집 수가 {}개일 때 각 군집별 시계열 분포'.format(n_cluster, fontsize=15, y=1.05))
    plt.axis('off')

    if algorithm=='kmeans':
        labels=df_with_labels['cluster(n={})'.format(n_cluster)]
    elif algorithm=='som':
        labels=df_with_labels.cluster

    idx=1
    for label in set(labels): # 각 군집에 대해
        if algorithm=='kmeans':
            df = train[train.num.isin(list(labels[labels==label].keys()))]
        elif algorithm=='som':
            df = train[train.num.isin(som_df[som_df.cluster==label].num.values())]
        
        hour = df.hour.unique()
        weekday = df.weekday.unique()

        ### x축: 시간 ###
        ax = fig.add_subplot(n_cluster, 2, idx); idx+=1
        ax.set(title='{}번 군집의 시간별 평균 전력사용량'.format(label))
        ax.set_xticks(hour)
        for num in df.num.unique():
            df_one=df[df.num==num]
            sns.lineplot(x=hour, y=df_one.groupby('hour')['전력사용량(kWh)'].mean(), color='grey', alpha=0.3)
        sns.lineplot(x=hour, y=df.groupby('hour')['전력사용량(kWh)'].mean(),color='red')

        ### x축: 요일 ###
        ax = fig.add_subplot(n_cluster, 2, idx);idx+=1
        ax.set(title='{}번 군집의 요일별 평균 전력 사용량'.format(label))
        for num in df.num.unique():
            df_one = df[df.num==num]
            df_one_counts=df_one.groupby('weekday')['전력사용량(kWh)'].mean()
            df_one_counts = df_one_counts.reindex(weekday)
            sns.lineplot(x=weekday,y=df_one_counts,color='grey',alpha=0.3)
        df_counts = df.groupby('weekday')['전력사용량(kWh)'].mean()
        df_counts = df_counts.reindex(weekday)
        sns.pointplot(x=weekday, y=df_counts, color='red')
        ax.legend('총 {}개 건물'.format(df.num.nunique()), loc=1)

    plt.show()