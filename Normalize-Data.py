import csv
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
from sklearn.ensemble import GradientBoostingRegressor as gBR
import time
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.model_selection import train_test_split

start_time = time.time()

train_df = pd.read_csv('train.csv')
train_df_dict = dict()

length = len(train_df)
# print train_df.iloc[:1, :]

for i in range(length):
     train_df_dict[train_df.loc[i, 'Id']] = train_df.loc[i, 'SalePrice']
     train_df.loc[i, 'SalePrice'] = math.log(float(train_df.loc[i, 'SalePrice']))

# print train_df.iloc[:1, :]

col = list(c for c in train_df.columns)
col = col[1: len(col) - 1]

for i in col:
     train_df.loc[:, i] = train_df.loc[:, i].astype('category')

categorical_columns = train_df.select_dtypes(['category']).columns
train_df[categorical_columns] = train_df[categorical_columns].apply(lambda x: x.cat.codes)

# print train_df.iloc[:1, :]

train_df = train_df.sample(frac=1).reset_index(drop=True)
# print train_df.iloc[:1, :]

train_df_vector = train_df.iloc[:, 1: len(col)]
train_df_target = train_df.iloc[:, len(col) + 1]

reg = gBR(loss='ls', n_estimators=1000)
reg = reg.fit(train_df_vector, train_df_target)

lr_predict = reg.predict(train_df_vector)
lr_accuracy = metrics.mean_absolute_error(train_df_target, lr_predict)
print("Time for the regressor to train and predict on the training data subset is := %.2f" % (time.time() - start_time))
print("Accuracy is := " + str(lr_accuracy))

