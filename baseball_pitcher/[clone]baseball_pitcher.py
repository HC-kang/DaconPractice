#####
# 5.1 문제 정의
###
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
print(matplotlib.__version__)
print(np.__version__)
print(pd.__version__)
print(sns.__version__)
print(sm.__version__)

import pandas as pd
atKbo_11_18_KboRegSsn = pd.read_csv('kbo_yearly_foreigners_2011_2018.csv')
atKbo_11_18_MlbTot = pd.read_csv('fangraphs_foreigners_2011_2018.csv')
atKbo_19_MlbTot = pd.read_csv('fangraphs_foreigners_2019.csv')

print(atKbo_11_18_KboRegSsn.shape)
print(atKbo_11_18_MlbTot.shape)
print(atKbo_19_MlbTot.shape)

print(atKbo_11_18_KboRegSsn.columns)
print(atKbo_11_18_MlbTot.columns)
print(atKbo_19_MlbTot.columns)

atKbo_11_18_KboRegSsn[['ERA', 'TBF']].hist()
print(atKbo_11_18_KboRegSsn[['ERA', 'TBF']].describe())

atKbo_11_18_MlbTot[['ERA', 'TBF']].hist()
print(atKbo_11_18_MlbTot[['ERA', 'TBF']].describe())

m_mean = (atKbo_11_18_MlbTot.groupby('pitcher_name')['ERA'].mean().reset_index().rename(columns = {'ERA':"MLB_mean"}))
k_mean = (atKbo_11_18_KboRegSsn.groupby('pitcher_name')['ERA'].mean().reset_index().rename(columns = {'ERA':'KBO_mean'}))

df = pd.merge(m_mean, k_mean, how = 'inner', on = 'pitcher_name')
df.head()

df.plot(kind = 'scatter', x = 'MLB_mean', y = 'KBO_mean')
print(df.corr())

atKbo_11_18_StatCast = pd.read_csv('baseball_savant_foreigners_2011_2018.csv')
atKbo_19_StatCast = pd.read_csv('baseball_savant_foreigners_2019.csv')

print(atKbo_11_18_StatCast.shape)
print(atKbo_19_StatCast.shape)

print(atKbo_19_StatCast.columns)
print(atKbo_19_StatCast.columns)

atKbo_11_18_StatCast[['events', 'description', 'pitch_name']]

(atKbo_11_18_StatCast['events'].value_counts().sort_values(ascending = True).plot(kind = 'barh', figsize = (8, 8)))

(atKbo_11_18_StatCast['description'].value_counts().sort_values(ascending = True).plot(kind = 'barh', figsize = (8, 8)))

(atKbo_11_18_StatCast['pitch_name'].value_counts().sort_values(ascending = True).plot(kind = 'barh', figsize = (8, 8)))


#####
# 5.3 데이터 전처리
###

## 5.3.1. 가설을 확인하기 위한 투수 집단 선정하기

import pandas as pd
import os 
os.chdir('/Users/heechankang/projects/pythonworkspace/dacon_data/baseball_pitcher')

# 데이터셋 불러오기
atKbo_11_18_KboRegSsn = pd.read_csv('kbo_yearly_foreigners_2011_2018.csv')
atKbo_11_18_MlbTot = pd.read_csv('fangraphs_foreigners_2011_2018.csv')
atKbo_11_18_StatCast = pd.read_csv('baseball_savant_foreigners_2011_2018.csv')
atKbo_19_MlbTot = pd.read_csv('fangraphs_foreigners_2019.csv')
atKbo_19_StatCast = pd.read_csv('baseball_savant_foreigners_2019.csv')

atKbo_11_18_KboRegSsn.head(10)

atKbo_19_MlbTot.head(10)

print('KBO: ', len(atKbo_11_18_KboRegSsn['pitcher_name'].unique()))
    # unique 함수를 활용해 KBO 선수들의 인원수 확인 - 62
print('MLB: ', len(atKbo_11_18_MlbTot['pitcher_name'].unique()))
    # 상동 - 13
print('StatCast: ', len(atKbo_11_18_StatCast['pitcher_name'].unique()))
    # 상동 - 60

target = (set(atKbo_11_18_KboRegSsn['pitcher_name']) &
          set(atKbo_11_18_MlbTot['pitcher_name']) &
          set(atKbo_11_18_StatCast['pitcher_name']))
          # set를 통해 중복을 제거하고 전체 인원을 하나의 세트로 만들기.

print(type(target)) # set

target = sorted(list(target)) # 편의를 위해 오름차순 정의 후 리스트로 변환

print(type(target)) # list

print(len(target)) # 중복제거 후 총 명단 - 57명

_idx = atKbo_11_18_KboRegSsn.query('pitcher_name in @target').\
       groupby('pitcher_name')['year'].idxmin().values
firstYearInKBO_11_18 = atKbo_11_18_KboRegSsn.loc[_idx, :]
firstYearInKBO_11_18.head()
    # 나같은 초보자는 이런 부분에서 배울 게 참 많다고 느낀다. 
    # @target으로 쿼리문 안에서 str형태로 명단을 불러내고, year필드의 가장 작은 인덱스를 뽑아서
    # 그 values만을 뽑아 내어 인덱스로 만들어준다.


## 5.3.2. 유효한 데이터 선정하기
TBF_median = firstYearInKBO_11_18['TBF'].median() # TBF: 상대한 타자 수
ERA_median = firstYearInKBO_11_18['ERA'].median() # ERA: 평균자책점
Elite_11_18 = firstYearInKBO_11_18.query('TBF >= @TBF_median & ERA <= @ERA_median')
Elite_11_18


## 5.4 모델 구축과 검증
### 5.4.1 선형 회귀 분석

# 자료 불러오기
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (10, 10))
sns.set_style('darkgrid')
sns.scatterplot(data = atKbo_11_18_StatCast.sort_values('pitch_name'),
                x = 'plate_x',
                y = 'plate_z',
                hue = 'pitch_name',
                alpha = 0.1)
plt.show()


# 지역 표시하기
plt.figure(figsize = (10, 10))
sns.set_style('darkgrid')
sns.scatterplot(data = atKbo_11_18_StatCast.sort_values('pitch_name'),
                x = 'plate_x',
                y = 'plate_z',
                hue = 'pitch_name',
                alpha = 0.1)
plt.plot([-1, -1], [1.5, 3.5], 'black')
plt.plot([1, 1], [1.5, 3.5], 'black')
plt.plot([-1, 1], [1.5, 1.5], 'black')
plt.plot([-1, 1], [3.5, 3.5], 'black')
plt.show()


# called strike
plt.figure(figsize = (10, 10))
sns.set_style('darkgrid')
sns.scatterplot(data = (atKbo_11_18_StatCast.
                        sort_values('pitch_name').
                        query('description == "called_strike"')),
                x = 'plate_x',
                y = 'plate_z',
                hue = 'pitch_name',
                alpha = 0.1)
plt.plot([-1, -1], [1.5, 3.5], 'black')
plt.plot([1, 1], [1.5, 3.5], 'black')
plt.plot([-1, 1], [1.5, 1.5], 'black')
plt.plot([-1, 1], [3.5, 3.5], 'black')
plt.plot()

lists = [1,3,5]
set(lists)
lists