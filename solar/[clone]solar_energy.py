import pandas as pd
import numpy as np
from datetime import timedelta
from glob import glob
import plotly.express as px

pd.options.plotting.backend = 'plotly'

# plotly.io를 import 한 후 renderers 기본값을 꼭 'notebook_connected'로 설정
import plotly.io as pio
pio.renderers.default = 'notebook_connected'
######################################################################
##### 이게 뭔지...? ####################################################
######################################################################

path = '/Users/heechankang/projects/pythonworkspace/dacon_data/solar_data/'
files = sorted(glob(path+'*.csv'))


site_info = pd.read_csv(files[4])   # 발전소 정보
energy = pd.read_csv(files[2])      # 발전소별 발전량

dangjin_fcst_data = pd.read_csv(files[0])   # 당진 예보 데이터
dangjin_obs_data = pd.read_csv(files[1])    # 당진 기상 관측 자료

ulsan_fcst_data = pd.read_csv(files[5])     # 울산 예보 데이터
ulsan_obs_data = pd.read_csv(files[6])      # 울산 기상 관측 자료

sample_submission = pd.read_csv(files[3])   # 제출 양식

site_info.head()
energy.head(20)
dangjin_fcst_data.head()
dangjin_obs_data.head()

site_info.head() # 패널 이름, 용량, 주소, 설치각도, 입사각, 위도 및 경도.
site_info.shape # shape (4, 7)

energy.shape    # shape (25632, 5)
energy.info()   # count 수를 보니 floating이랑 warehouse에 좀 누락이 있음.
energy.head(20) # 9시부터 발전 시작.
# 당진이 상대적으로 가장 많이 발전 중.
site_info 
# 당진(수상) : 1.0, 당진(창고) : 0.7, 당진 : 0.7, 울산 : 0.5

energy.describe()
# 수상과 창고 2종은 결측치가 좀 있음.
energy.isnull().sum() # 각 24건, 48건

energy['date'] = energy['time'].apply(lambda x: x.split()[0])
energy['time'] = energy['time'].apply(lambda x: x.split()[1])
energy['time'] = energy['time'].str.rjust(8,'0')
# 한자릿수 시간 앞에 0 추가해서 1시 -> 01시로 바꿔주기

# 24시를 00시로 바꿔주기
# @안바꿔주면 datetime형으로 바뀌지 않음@
energy.loc[energy['time']=='24:00:00', 'time'] = '00:00:00'
energy['time']=energy['date'] + ' ' + energy['time']
energy['time'] = pd.to_datetime(energy['time'])
energy.loc[energy['time'].dt.hour==0, 'time'] += timedelta(days=1)
        # 24:00:00->00:00:00으로 바뀌면서 일자 1일씩 더하기
energy


# '18년 3월 1주차 발전량 확인
fig = px.line(energy[:24*7], x='time', y=['dangjin_floating', 'dangjin_warehouse', 'dangjin', 'ulsan'])
fig.show()
# 4일이 낮긴 한데 왜낮은지 모르겠네....날씨도 이상한건 아닌듯?
dangjin_obs_data[2*24:4*24:]
ulsan_obs_data[2*24:4*24:]
# 이부분은 좀 더 확인이 필요할듯

energy['month']=energy['time'].dt.month
energy['hour'] =energy['time'].dt.hour
energy.head()


# 월별 발전량 확인
mean_month = energy.groupby('month').mean()
fig=px.bar(mean_month, x=mean_month.index, y=['dangjin_floating', 'dangjin_warehouse', 'dangjin', 'ulsan'])
fig.show()
# 여름인 7~8월에 발전량이 오히려 봄보다 적은건 신기하긴 함.
# 판넬 입사각이 2~30도라서 그런건 모르겠고, 아마 장마 때문인가?

mean_hour = energy.groupby('hour').mean()
fig = px.bar(mean_hour, x=mean_hour.index, y = ['dangjin_floating', 'dangjin_warehouse', 'dangjin', 'ulsan'])
fig.show()
# 저녁 8시부터 다음날 06시까지는 발전량 0


# 발전소별 발전량 분포
cols = ['dangjin_floating', 'dangjin_warehouse', 'dangjin', 'ulsan']
fig = px.box(energy[energy[cols]!=0], x=cols)
fig.update_traces(quartilemethod='exclusive')
fig.show()
site_info



# 당진 기상 관측자료

dangjin_obs_data.info()
dangjin_obs_data.isnull().sum()
# 기온, 풍속, 풍향, 습도 약 30건씩 결측, 전운량 3970건 결측. 맑아서?? 흠..
dangjin_obs_data[24*152:24*154]
# 맑아서 빠진게 아니라 그냥 기록을 안한거였네,, '18년 8월 1일 01:00부터 기록 시작함.

# 기온, 풍속, 풍향, 습도는 그냥 이전걸로 ffill 하면 될텐데....... 전운량은 답이 안나오네

# 일단은 따라가기.
def weather_preprocessing(df):
    df['전운량(10분위)'].fillna(0, inplace=True)
    df.fillna(method = 'ffill', inplace=True)
    df['일시']=pd.to_datetime(df['일시'])
    return df

dangjin_obs_data = weather_preprocessing(dangjin_obs_data)
dangjin_obs_data.isnull().sum()


# 발전량과 날씨의 상관관계 (당진)
df_joined = pd.merge(energy, dangjin_obs_data, left_on='time', right_on='일시', how = 'inner')
cols = ['time', 'dangjin_floating', 'dangjin_warehouse', 'dangjin', '기온(°C)', '풍속(m/s)', '풍향(16방위)', '습도(%)', '전운량(10분위)']
dangjin = df_joined[cols]
dangjin
df_joined
dangjin_obs_data
energy

dangjin.corr()

fig = px.imshow(dangjin.corr())
fig.show()


# 울산
ulsan_obs_data.info()
ulsan_obs_data.isnull().sum()
# 기온 4, 풍속, 풍향, 습도 1, 전운량 또 825개 결측
ulsan_obs_data[24*30:24*32]
# 여기도 그냥 맑아서 결측은 아닌듯. 다만 주로 야간이라 그냥 무시해도 문제는 없을듯,,

ulsan_obs_data = weather_preprocessing(ulsan_obs_data)

df_joined = pd.merge(energy, ulsan_obs_data, left_on='time', right_on='일시', how = 'inner')
cols = ['time', 'ulsan', '기온(°C)', '풍속(m/s)', '풍향(16방위)', '습도(%)' ,'전운량(10분위)']
ulsan = df_joined[cols]
ulsan.corr()

# 히트맵
fig = px.imshow(ulsan.corr())
fig.show()

sample_submission.head()
sample_submission.tail()