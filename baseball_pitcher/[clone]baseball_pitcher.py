#####
# 5.1 문제 정의
###
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import os
print(matplotlib.__version__)
print(np.__version__)
print(pd.__version__)
print(sns.__version__)
print(sm.__version__)
os.chdir('/Users/heechankang/projects/pythonworkspace/dacon_data/baseball_pitcher')

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

edgePitches = (atKbo_11_18_StatCast.query(
              '(plate_x >= 0.8 & plate_x <= 1.2 & plate_z <= 3.7 & plate_z >= 1.3) |\
               (plate_x <= -0.8 & plate_x >= -1.2 & plate_z <=3.7 & plate_z >=1.3)|\
               (plate_x >= -0.8 & plate_x <= 0.8 & plate_z <= 1.7 & plate_z >=1.3)|\
               (plate_x >= -0.8 & plate_x <= 0.8 & plate_z <= 3.7 & plate_z >= 3.3)').
               query('pitch_name.notnull()', engine = 'python').
               query('description == "called_strike"'))

plt.figure(figsize = (10, 10))
sns.set_style('darkgrid')
sns.scatterplot(data = edgePitches,
                x = 'plate_x',
                y = 'plate_z',
                hue = 'pitch_name',
                alpha = 0.1)
plt.plot([-1, -1], [1.5, 3.5], 'black')
plt.plot([1, 1], [1.5, 3.5], 'black')
plt.plot([-1, 1], [1.5, 1.5], 'black')
plt.plot([-1, 1], [3.5, 3.5], 'black')
plt.show()


(edgePitches[['pitcher_name', 'pitch_name', 'game_date']].
    groupby(['pitcher_name', 'pitch_name']).
    count().
    head(10)
)

(edgePitches[['pitcher_name', 'pitch_name', 'game_date']].
    groupby(['pitcher_name', 'pitch_name']).
    count().
    groupby('pitcher_name').
    apply(lambda x: x / x.sum()).
    head(10)
)

(edgePitches[['pitcher_name', 'pitch_name', 'game_date']].
    groupby(['pitcher_name', 'pitch_name']).
    count().
    groupby('pitcher_name').
    apply(lambda x: x/x.sum()).
    query('game_date >= 0.1').
    head(10)
)

coordEdge = (edgePitches[['pitcher_name', 'pitch_name', 'game_date']].
    groupby(['pitcher_name', 'pitch_name']).
    count().
    groupby('pitcher_name').
    apply(lambda x:x/x.sum()).
    query('game_date >= 0.1').
    groupby('pitcher_name').
    count()
)

coordEdge = coordEdge.reset_index().rename(columns = {'game_date':'num_pitches'})

coordEdge.head()

Elite_11_18 = Elite_11_18.reset_index()

Elite_11_18 = Elite_11_18.merge(coordEdge, on = 'pitcher_name')

Elite_11_18.boxplot('ERA', 'num_pitches')

import statsmodels.api as sm

y = Elite_11_18.ERA.values
X = sm.add_constant(Elite_11_18.num_pitches.values)

model = sm.OLS(y, X)

result = model.fit()

result.summary()


#####
# 아웃확률 추정하기
###

atKbo_11_18_StatCast[['batter', 'events', 'description']].head(10)

(atKbo_11_18_StatCast[['batter', 'events', 'description']].
    query('events.notnull()', engine = 'python').
    head(10)
)

def recordInning(key, dic):    
    if dic.get(key) == None :
        dic[key] = 1
    else :
        dic[key] += 1
    
    return dic
    

def getInningResult(df):
    batterCount = 0
    batterCountTemp = 0
    outs = ['out', 'out', 'out']
    inningDict = {}
    
    for idx in range(len(df)-1, -1, -1):
        batterCount += 1
        
        if 'out' in df.events.iloc[idx]:
            outs.pop()
        
        # out이 3번 나오면 기록
        if len(outs) == 0:
            _key = f'I_{batterCount - batterCountTemp}'
            inningDict = recordInning(_key, inningDict)
            batterCountTemp = batterCount
            
            if idx != 0 :
                outs = ['out', 'out', 'out']
            
    if len(outs) != 0:
        _key = f'I_{batterCount - batterCountTemp + len(outs)}'
        inningDict = recordInning(_key, inningDict)

    return pd.DataFrame(data = dict(sorted(inningDict.items())), index = [0])


MLB_11_18_InningSummary = (atKbo_11_18_StatCast.
    query('events.notnull()', 
          engine = 'python').
    groupby(['pitcher_name', 'game_date']).
    apply(getInningResult))

MLB_11_18_InningSummary.head()

MLB_11_18_InningSummary = (MLB_11_18_InningSummary.
    groupby('pitcher_name').
    sum()[sorted(MLB_11_18_InningSummary.columns)])

MLB_11_18_InningSummary

MLB_11_18_InningSummary = MLB_11_18_InningSummary.reset_index()

Elite_11_18_InningSummary = (MLB_11_18_InningSummary.
    query('pitcher_name in @Elite_11_18.pitcher_name').
    reset_index(drop = True))

Elite_11_18_InningSummary

def makeC1(df):
    '''
    Parameters:
    -----------------

    Returns:
    -----------------
    pd.Sereis
        논문에성 정의한 C1값
    '''
    return df.sum(axis = 1)

def makeC2(df):
    '''
    Parameters:
    -----------------

    Returns:
    -----------------
    pd.Sereis
        논문에성 정의한 C@값
    '''
    return 3*(df['I_3'] + df['I_4'])

def makeC3(df):

    '''
    Parameters:
    -----------------

    Returns:
    -----------------
    pd.Sereis
        논문에성 정의한 C3값
    '''
    output = 0
    for N in range(5, 18):
        try:
            output += (N-3)*df[f'I_{N}']
        except:
            continue
    return output

def makeDelta(df):
    '''
    Parameters:
    ---------
    df: InningSummary with C1, C2, C3

    Returns:
    --------
    pd.Series
        논문에서 정의한 Delta값
    '''
    Delta = ((-df['C1'] + df['C2'] + 2*df['C3'])+
            ((df['C1'] - df['C2'] - 2*df['C3']).pow(2)+
            4*df['C3']*(3*df['C1'] + df['C2'] + 3*df['C3'])).pow(0.5))/\
            (2*(3*df['C1'] + df['C2'] + 3*df['C3']))
    return Delta                

def makeOutProb(df):
    '''
    Parameters:
    ----------
    df: InningSummary

    Returns:
    --------
    pd.DataFrame
        InningSummary iwth C1, C2, C3, Delta, outProb
    '''
    df['C1'] = makeC1(df)
    df['C2'] = makeC2(df)
    df['C3'] = makeC3(df)

    df['Delta'] = makeDelta(df)
    df['outProb'] = 1 - df['Delta']

    return df

Elite_11_18_InningSummary = makeOutProb(Elite_11_18_InningSummary)
Elite_11_18_InningSummary.sort_values('outProb', ascending = False)

edgePitches_19 = \
(atKbo_19_StatCast.query(
    '(plate_x >= 0.8 & plate_x <= 1.2 & plate_z <= 3.7 & plate_z >= 1.3) | \
     (plate_x <= -0.8 & plate_x >= -1.2 & plate_z <= 3.7 & plate_z >= 1.3) | \
     (plate_x >= -0.8 & plate_x <= 0.8 & plate_z <= 1.7 & plate_z >= 1.3) | \
     (plate_x >= -0.8 & plate_x <= 0.8 & plate_z <= 3.7 & plate_z >= 3.3)').
 query('pitch_name.notnull()', engine='python').
 query('description == "called_strike"'))

coordEdge_19 = \
(edgePitches_19[['pitcher_name', 'pitch_name', 'game_date']].
 groupby(['pitcher_name', 'pitch_name']).
 count().
 groupby('pitcher_name').
 apply(lambda x : x / x.sum()).
 query('game_date >= 0.1').
 groupby('pitcher_name').
 count())

MLB_19_InningSummary = (atKbo_19_StatCast.query('events.notnull()', engine = 'python').
                        groupby(['pitcher_name', 'game_date']).
                        apply(getInningResult))

MLB_19_InningSummary = (MLB_19_InningSummary.
                        groupby('pitcher_name').
                        sum()[sorted(MLB_19_InningSummary.columns)])

MLB_19_InningSummary = MLB_19_InningSummary.reset_index()

MLB_19_InningSummary = makeOutProb(MLB_19_InningSummary)
MLB_19_InningSummary.sort_values('outProb', ascending=False)

