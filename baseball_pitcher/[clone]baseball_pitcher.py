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
atKbo_11_18_KboRegSsn
atKbo_11_18_KboRegSsn.query('pitcher_name in @target').groupby('pitcher_name')
_idx = atKbo_11_18_KboRegSsn.query('pitcher_name in @target').groupby('pitcher_name')['year'].idxmin().values
firstYearInKBO_11_18 = atKbo_11_18_KboRegSsn.loc[_idx, :]
firstYearInKBO_11_18.shape
firstYearInKBO_11_18.head()