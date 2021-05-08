import os
os.chdir('/Users/heechankang/projects/pythonworkspace/dacon_data/jeju_citrus_data')

import pandas as pd

# 데이터 로드
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
bts = pd.read_csv('bus_bts.csv')
jeju_life = pd.read_csv('jeju_financial_life_data.csv')
weather = pd.read_csv('weather.csv', encoding='cp949')
rain = pd.read_csv('rain.csv', encoding='utf-8')

# 지하철 데이터 형태 확인
train.head()

# 지하철 데이터 정보 확인
train.info()
train.shape
train.describe()

# 버스 데이터 실물 확인
bts.head()

# 버스 데이터 정보 확인
bts.info()
bts.shape
bts.describe()
bts.info(null_counts = True)

# jeju_financial_life_data 실물 확인
jeju_life.head()

# jeju_life 정보 확인
jeju_life.info() 
jeju_life.shape
jeju_life.describe()
    # zip code 가 왜 int?
    # 일자도 int로 되어있음. 
    # 성별이 int로 되어있음.

# weather : 9월 1일 ~30일 제주도 전체의 오전10시 기상정보 데이터.
weather.head()

# weather 정보 확인
weather.info()
weather.shape
weather.describe()
    # 강수량이 왜 object? 날짜도 object네

# rain 실자료 확인
rain.head()

# rain 정보 확인
rain.info()
    # 일시가 object네.
    # 강수량에 결측치 118개
rain.shape
rain.describe()

#####
# 데이터 시각화를 통한 탐색적 데이터 분석
###

# 시각화에 필요한 라이브러리 불러오기
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# 내부에 결과를 출력하도록 설정
%matplotlib inline

# 시각화 한글 폰트 설정
# 윈도우
# plt.re('font', family='Malgun Gothic')
# Mac
plt.rc('font', family='AppleGothic')

# 마이너스 기호 출력
plt.rc('axes', unicode_minus=False)

# 분석에 문제가 없는 경고메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

# ------------- 데이터를 변형할 필요성도 있으므로 별도 데이터로 복사해 활용
# 데이터 복사본 생성
traindata = train.copy()
traindata.head()
# 타깃 변수(퇴근시간 승차인원) 분포 확인하기
# 타깃값이 특정 구간에 치우쳐 있는지, 분포 범위 등을 확인하기 위함임.
# 히스토그램 활용 시각화 

sns.distplot(traindata['18~20_ride'], kde =False, bins = 50)
plt.axis([0, 50, 0, 450000]) # [x축 최소, 최댓값, y축 최소, 최댓값]
plt.title('퇴근 시간 승차 인원 히스토그램') # 그래프 제목 지정
plt.show()

# 타깃 변수(퇴그시간 승차인원) 분포 확인
traindata.groupby('18~20_ride').size().head(10)


# 요일에 따른 퇴근시간 평균 탑승객 수

# 요일 변수 생성
# 날짜형으로 변환
traindata['date'] = pd.to_datetime(traindata['date'])
# 요일을 문자형으로 추출해 변수 생성
traindata['weekday'] = traindata['date'].dt.strftime('%a')
traindata.head()

# 요일별 퇴근시간 평균 탑승객 수 시각화
sns.barplot(x='weekday', y='18~20_ride', data=traindata)
plt.title('요일에 따른 퇴근시간 평균 탑승객 수')
plt.show()

# 버스 종류에 따른 탑승객 수
sns.barplot(x='in_out', y = '18~20_ride', data = traindata)
plt.title('버스 종류에 따른 평균 탑승객 수')
plt.show()
    # 시외 2명, 시내 1.3명

# 일별 출퇴근 시간의 총 승차 인원 데이터 생성
traindata['8~10_ride'] = traindata['8~9_ride'] + traindata['9~10_ride']
eda_data=traindata.groupby('date')[['18~20_ride', '8~10_ride']].agg('sum').reset_index()
eda_data.head()

# 일별 출퇴근 시간 탑승객 수 시각화
plt.plot('date', '18~20_ride', 'g-', label='퇴근 시간 탑승객 수', data = eda_data)
plt.plot('date', '8~10_ride', 'b-', label='출근 시간 탑승객 수', data= eda_data)
plt.gcf().autofmt_xdate() # x축의 라벨이 서로 겹치지 않도록 설정
plt.legend(loc=0) # 그래프 상에서 최적의 위치에 범례 표시
plt.title('일별 출퇴근 시간 탑승객 수')
plt.show()



#####
# 데이터 전처리
###
# 학습 데이터와 테스트 데이터를 구분하기 위한 변수 생성
train['cue'] = 0
test['cue'] = 1
train
test


# 학습 데이터와 테스트 데이터 통합
df = pd.concat([train, test], axis = 0)
df

# 요일 만들기
# datetime 변수형으로 변환
df['date']= pd.to_datetime(df['date'])

# 요일 추출(0-월요일~6-일요일)
df['weekday'] = df['date'].dt.weekday
df[['weekday']].head()  # 이와중에 DataFrame 유지하려고 대괄호 2겹.
df['weekday'].head()

# 요일별 평균 탑승객 수
def week_mean() :
    # 전체 데이터에서 train 데이터에 해당하는 행 추출
    train_data = df.query('cue==0').reset_index(drop=True)
    
    # 일괄적으로 1의 값을 가지는 'weekdaymean'변수 생성
    df['weekdaymean'] = 1

    # 각 요일에 해당하는 인덱스 추출
    index0 = df.query('weekday==0').index
    index1 = df.query('weekday==1').index
    index2 = df.query('weekday==2').index
    index3 = df.query('weekday==3').index
    index4 = df.query('weekday==4').index
    index5 = df.query('weekday==5').index
    index6 = df.query('weekday==6').index

    # 인덱스를 활용하여 'weekdaymean'의 값을 각 요일에 맞는 평균 탑승 승객수로 변경 
    df.iloc[index0,-1] = train_data.query('weekday==0')['18~20_ride'].mean()
    df.iloc[index1,-1] = train_data.query('weekday==1')['18~20_ride'].mean()
    df.iloc[index2,-1] = train_data.query('weekday==2')['18~20_ride'].mean()
    df.iloc[index3,-1] = train_data.query('weekday==3')['18~20_ride'].mean()
    df.iloc[index4,-1] = train_data.query('weekday==4')['18~20_ride'].mean()
    df.iloc[index5,-1] = train_data.query('weekday==5')['18~20_ride'].mean()
    df.iloc[index6,-1] = train_data.query('weekday==6')['18~20_ride'].mean()
    
    return df

df
df = week_mean()
df.head()
df[['weekdaymean']].head()


# 버스 종류별 평균 탑승객 수
def inout_mean():
    # 전체 데이터에서 train 데이터에 해당하는 행 추출
    train_data = df.query('cue==0').reset_index(drop=True)

    # 일괄적으로 1의 값을 가지는 in_out_mean 변수 생성
    df['in_out_mean'] = 1

    # 버스 종류별 인덱스 추출
    in_index = df.query('in_out == "시내"').index
    in_index = df.query('in_out == "시외"').index

    # 인덱스를 활용하여 in_out_mean의 값을 각 버스 종류에 맞는 평균 탑승 승객 수로 변경
    df.iloc[in_index, -1] = train_data.query('in_out == "시내"')['18~20_ride'].mean()
    df.iloc[in_index, -1] = train_data.query('in_out == "시외"')['18~20_ride'].mean()

    return df

# 함수를 실행하여 변수 생성
df = inout_mean()
df.head()
df['in_out_mean'].head()



# 일별 오전 시간대의 총 탑승객 수
# 날짜별 오전 시간에 탑승한 총 탑승객 수
f = df.groupby('date')['6~7_ride', '7~8_ride', '8~9_ride', '9~10_ride', '10~11_ride'].sum().reset_index()
# 변수명 변경
f.columns = ['date','6~7_ride_sum', '7~8_ride_sum', '8~9_ride_sum', '9~10_ride_sum', '10~11_ride_sum']
# 기존 데이터프레임에 새로운 변수를 병합
df = pd.merge(df, f, how = 'left', on='date')

f.head()

df[['date', '6~7_ride_sum', '7~8_ride_sum', '8~9_ride_sum', '9~10_ride_sum', '10~11_ride_sum']].head()

