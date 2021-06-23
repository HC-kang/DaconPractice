#Library Imports
import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

#######딥러닝 라이브러리##########
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape, GRU, RNN

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

import tensorflow.keras
from keras.layers import Dense
from tensorflow.keras.models import Sequential
from keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers.convolutional import MaxPooling1D,MaxPooling2D
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import TimeDistributed, Dense, Flatten,ConvLSTM2D,RepeatVector
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import tensorflow as tf 
from tensorflow.keras import backend as k 


from sklearn.preprocessing import MinMaxScaler

os.chdir('/Users/heechankang/projects/pythonworkspace/dacon_data/energy_consume')
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

tf.keras.backend.set_floatx('float64')

train = pd.read_csv('train.csv', encoding='cp949')
test = pd.read_csv('test.csv', encoding='cp949')
sample_submission = pd.read_csv('sample_submission.csv', encoding='cp949')

#train.shape 122400 X 10
#60개의 건물 X 85일 24시간 =122400
train

#test.shape 10080 X 9
#60개의 건물 X 7일 24시간 =10080
test


input_window =996 #임의의 수
output_window = 24 #168 7일 24시간
window = 12 #window는 12시간 마다는 12시간 마다
num_features = 1 #베이스라인은 feature를 하나만 사용했습니다.
num_power = 60
end_=168
lstm_units=32
dropout=0.2
EPOCH=30
BATCH_SIZE=128

#train을 tensor로 변경 (60, 24*85, 1)
train_x=tf.reshape(train.iloc[:,2].values, [num_power, 24*85, num_features])
print(f'train_x.shape:{train_x.shape}')

#train_window_x np.zeros를 만듬 (60, 85, 996, 1)
train_window_x= np.zeros(( train_x.shape[0], (train_x.shape[1]-(input_window + output_window))//window, input_window, num_features)) 
train_window_y= np.zeros(( train_x.shape[0], (train_x.shape[1]-(input_window + output_window))//window, output_window, num_features))
print(f'train_window_x.shape:{train_window_x.shape}')
print(f'train_window_y.shape:{train_window_y.shape}')

#train_window_x에 train값 채워넣기
for example in range(train_x.shape[0]):
    
    for start in range(0, train_x.shape[1]-(input_window+output_window), window):
        end=start+input_window
        train_window_x[example, start//window, :] = train_x[example, start: end               , :]
        train_window_y[example, start//window, :] = train_x[example, end  : end+ output_window, :]

#new_train_x, reshape통해 lstm에 알맞은 형태로 집어넣기
new_train_x=tf.reshape(train_window_x, [-1, input_window, num_features])
new_train_y=tf.reshape(train_window_y, [-1, output_window,num_features])
print(f'new_train_x.shape:{new_train_x.shape}')
print(f'new_train_y.shape:{new_train_y.shape}')

#####층 쌓기###########


model=Sequential([
LSTM(lstm_units, return_sequences=False, recurrent_dropout=dropout),
Dense(output_window * num_features, kernel_initializer=tf.initializers.zeros()), 
Reshape([output_window, num_features])
])
X_train_ma = X_train_ma.reshape(X_train_ma.shape[0],1,1, X_train_ma.shape[2])
X_valid_ma = X_valid_ma.reshape(X_valid_ma.shape[0],1,1, X_valid_ma.shape[2])
X_test_ma = X_test_ma.reshape(X_test_ma.shape[0],1,1, X_test_ma.shape[2])
n_steps = X_train_ma.shape[0]
n_steps = 1
n_features = X_train_ma.shape[2]
#####층 쌓기###########

model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu'), input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2,activation='relu')))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(64, activation='relu',return_sequences=True)))
model.add(Bidirectional(LSTM(64, activation='relu')))
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse',metrics = ['mae','mse','mape'])

history = model.fit(X_train_ma, y_train, epochs=100, batch_size=30, validation_data=(X_valid_ma, y_valid), verbose=1)



#######Compile 구성하기################


model.compile(optimizer='rmsprop', loss='mae', metrics=['mae'])
# 에포크가 끝날 때마다 점(.)을 출력해 훈련 진행 과정을 표시합니다
class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 10 == 0: print('')
        print('.', end='')

#가장 좋은 성능을 낸 val_loss가 적은 model만 남겨 놓았습니다.
save_best_only=tf.keras.callbacks.ModelCheckpoint(filepath="lstm_model.h5", monitor='val_loss', save_best_only=True)


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

#검증 손실이 10epoch 동안 좋아지지 않으면 학습률을 0.1 배로 재구성하는 명령어입니다.
reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)

######################
model.fit(new_train_x, new_train_y, epochs=EPOCH, batch_size=BATCH_SIZE, validation_split = 0.2, verbose=0,
          callbacks=[PrintDot(), early_stop, save_best_only , reduceLR])

model.summary()

#######################
prediction=np.zeros((num_power, end_, num_features))
new_test_x=train_x

for i in range(end_//output_window):
    start_=i*output_window
    next_=model.predict(new_test_x[ : , -input_window:, :])
    new_test_x = tf.concat([new_test_x, next_], axis=1)
    print(new_test_x.shape)
    prediction[:, start_: start_ + output_window, :]= next_
prediction =prediction *size + mini

submission['answer']=prediction.reshape([-1,1])
submission

submission.to_csv('baseline_submission1.csv', index=False)