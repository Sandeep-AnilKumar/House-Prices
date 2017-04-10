import pandas as pd
import math
from sklearn.ensemble import GradientBoostingRegressor as gBR
import time
from sklearn import metrics
import csv

start_time = time.time()

train_df = pd.read_csv('train.csv')
train_df_dict = dict()

length = len(train_df)

for i in range(length):
     train_df_dict[train_df.loc[i, 'Id']] = train_df.loc[i, 'SalePrice']
     train_df.loc[i, 'SalePrice'] = math.log10(float(train_df.loc[i, 'SalePrice']))

col = list(c for c in train_df.columns)
col = col[1: len(col) - 1]

for i in col:
     train_df.loc[:, i] = train_df.loc[:, i].astype('category')

categorical_columns = train_df.select_dtypes(['category']).columns
train_df[categorical_columns] = train_df[categorical_columns].apply(lambda x: x.cat.codes)


train_df = train_df.sample(frac=1).reset_index(drop=True)

train_df_vector = train_df.iloc[:, 1: len(col)]
train_df_target = train_df.iloc[:, len(col) + 1]

reg = gBR(loss='ls', n_estimators=10000)
reg = reg.fit(train_df_vector, train_df_target)

lr_predict = reg.predict(train_df_vector)
lr_accuracy = metrics.mean_absolute_error(train_df_target, lr_predict)
print("Time for the regressor to train and predict on the training data subset is := %.2f" % (time.time() - start_time))
print("Accuracy is := " + str(lr_accuracy))

start_time = time.time()
test_df = pd.read_csv('test.csv')
test_data_ids = list(test_df['Id'])

length = len(test_df)

col = list(c for c in test_df.columns)
col = col[1:]

for i in col:
     test_df.loc[:, i] = test_df.loc[:, i].astype('category')

categorical_columns = test_df.select_dtypes(['category']).columns
test_df[categorical_columns] = test_df[categorical_columns].apply(lambda x: x.cat.codes)

test_df_vector = test_df.iloc[:, 1: len(col)]

reg = gBR(loss='ls', n_estimators=10000)
reg = reg.fit(train_df_vector, train_df_target)

lr_predict = reg.predict(test_df_vector)
print("Time for the regressor to train and predict on the testing data subset is := %.2f" % (time.time() - start_time))

csv_file = open("submissions.csv", 'w')
wr = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_NONE)
wr.writerow(['Id', 'SalePrice'])

for index in range(0, len(test_data_ids)):
    wr.writerow([test_data_ids[index], math.pow(10, lr_predict[index])])
    index += 1
print("Done with predicting Sale Price values for the test data")
csv_file.close()

