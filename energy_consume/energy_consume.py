import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
# from glob import glob
import plotly.express as px
import os

os.chdir('/Users/heechankang/projects/pythonworkspace/dacon_data/energy_consume')

train = pd.read_csv('train.csv', encoding='cp949')
test = pd.read_csv('test.csv', encoding='cp949')
sample_submission = pd.read_csv('sample_submission.csv', encoding='cp949')

train.head()
train.tail()
train.shape
train.info()
train.describe()
train['date_time']=pd.to_datetime(train['date_time'])
train

test.head(10)
test.tail()
test.shape
test.info()
test.describe()
test['date_time']=pd.to_datetime(test['date_time'])

sample_submission.head()