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
import tqdm
import gc

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
group.drop(['wind', 'rain', 'cooling', 'solar','month', 'day', 'hour','num_day'], axis = 1, inplace=True)
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
def lag_feature(df, lags, col):
    tmp = df[['num_day','num','hour',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['num_day','num','hour', col+'_lag_'+str(i)]
        shifted['num_day'] += i
        df = pd.merge(df, shifted, on=['num_day','num','hour'], how='left')
    return df

train = lag_feature(train, [1,2,3,4,5,6,7,14,21], 'energy')

train[train['num']==60].head(40)
train[train['num']==60].tail(40)

train

train.isna().sum()
train.columns
train.info()

###################
# 일단 대략적으로 끝
###################
# ---*---*---* 첫 번째 모델 ---*---*---*
# 불쾌지수 미적용
train.to_pickle('full_train_nan.pkl')
train = pd.read_pickle('full_train_nan.pkl')

train.drop('date_time', axis = 1, inplace=True)

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

#########################
#########################
#########################

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
from xgboost import plot_importance
fig, ax = plt.subplots(1, 1, figsize = (10, 14))
plot_importance(model, ax = ax)

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

'''
피클 불러오자마자 난 그대로 두고 xg부스트 돌려보기. 생각보다 점수 잘 나옴. 
점수 : 22.7423089706
등수 : 제출 시 219등
'''
###############################
###############################
###############################
###############################

#########################
# ---*---*---*  ---*---*---*
# 두 번째 모델
# 불쾌지수 추가 스케일러 적용
from sklearn.preprocessing import MinMaxScaler

mmscaler = MinMaxScaler()

train_mmscaled = mmscaler.fit_transform(train)

#TODO:




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
    #('elasticnet_pipeline', elasticnet_pipeline), 
    #('poly_pipeline', poly_pipeline)
]


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
voting_pred= voting_pred.clip(0, 20)