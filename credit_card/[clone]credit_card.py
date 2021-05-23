import os
os.chdir('/Users/heechankang/projects/pythonworkspace/dacon_data/credit_card')

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('funda_train.csv')
submission = pd.read_csv('submission.csv')

train.shape
train.head()
train.info( )
train.isnull().sum()
train.isnull()
train.describe()

submission.shape
submission.head()
submission.info()
submission.describe()

train[train['amount']<0]

import rpy2

rpy2.__version__