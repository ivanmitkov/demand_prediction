# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:10:32 2021

@author: ivan.mitkov
"""
import os, sys
import pandas as pd
from numpy import argmax
from keras.utils import to_categorical
import numpy as np
from datetime import timedelta
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
from keras.callbacks import EarlyStopping
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_squared_log_error
from numpy.random import seed
import matplotlib.pyplot as plt
from datetime import datetime
import time
from sklearn.preprocessing import StandardScaler

os.chdir(r'your/local/repo')

df = pd.read_csv(r"task.csv")

#%% set the corresponding columns to the expected dtype
#df['date'] = df['date'].dt.strftime("%Y-%m-%d")
df['date'] = pd.to_datetime(df['date'])

#%% set the target variablevalue and date for the coresponding days ahead prediction
df.sort_values(by=['location', 'date'], inplace = True)
df_stacked = pd.DataFrame()
for dayahead in range(1, 8):
    df['days_ahead'] = dayahead
    df['date_shifted_obeservations'] = df.groupby('location')['date'].shift(periods = - dayahead)
    df['target_date'] = df['date'] + timedelta(days = dayahead)
    df['target_value'] = np.where(df['date_shifted_obeservations'] != df['target_date'], 999999, df.groupby('location')['sales'].shift(periods = - dayahead))
    df_stacked = df_stacked.append(df)

# %% ASSUMPTION: the chain knows when the restauraint will be closed and does not make any orders
df_stacked = df_stacked[df_stacked['target_value'] != 999999].reset_index(drop = True)

#%% feature engineering in order to extract time variantion per location,month, season, day of the week ect
df_stacked['dayofweek'] = df_stacked['date'].dt.dayofweek
df_stacked['target_date_dayofweek'] = df_stacked['target_date'].dt.dayofweek
df_stacked['season'] = df_stacked['target_date'].dt.month%12 // 3 + 1
df_stacked['sales_t-1'] = df_stacked.groupby('location')['sales'].shift(periods = 1)
df_stacked['sales_t-2'] = df_stacked.groupby('location')['sales'].shift(periods = 2)
df_stacked['sales_t-3'] = df_stacked.groupby('location')['sales'].shift(periods = 3)
df_stacked['mean(t-1_7)']=df_stacked.groupby('location')['sales'].transform(lambda s: s.rolling(7, min_periods=1).mean().shift().bfill())
df_stacked['std(t-1_7)']=df_stacked.groupby('location')['sales'].transform(lambda s: s.rolling(7, min_periods=1).std().shift().bfill())
df_stacked['mean(t-1_14)']=df_stacked.groupby('location')['sales'].transform(lambda s: s.rolling(14, min_periods=1).mean().shift().bfill())
df_stacked['std(t-1_14)']=df_stacked.groupby('location')['sales'].transform(lambda s: s.rolling(14, min_periods=1).std().shift().bfill())
df_stacked['mean(t-1_30)']=df_stacked.groupby('location')['sales'].transform(lambda s: s.rolling(30, min_periods=1).mean().shift().bfill())
df_stacked['std(t-1_30)']=df_stacked.groupby('location')['sales'].transform(lambda s: s.rolling(30, min_periods=1).std().shift().bfill())
df_stacked['mean_for_this_day_of_the_week'] = df_stacked.groupby(['location', 'dayofweek'])['sales'].transform(lambda s: s.shift().expanding().mean())
df_stacked['std_for_this_day_of_the_week'] = df_stacked.groupby(['location', 'dayofweek'])['sales'].transform(lambda s: s.shift().expanding().std())
df_stacked['distance'] = df['target_value'] - df['sales']
df_stacked['diff_shifted_1'] = df_stacked.groupby(['location', 'days_ahead', 'dayofweek'])['distance'].shift(periods = 1)
df_stacked['diff_shifted_2']=df_stacked.groupby(['location', 'days_ahead', 'dayofweek'])['distance'].shift(periods = 1)

#%% fill nans with extrem value. nans are caused by shifting values
df_stacked = df_stacked.fillna(999999)

#%% standardize and get predictor columns
dropcolumns = ['sales', 'season', 'location', 'date', 'target_value', 'date_shifted_obeservations', 'target_date', 'distance']
all_cols = list(df_stacked.columns)
norm_cols = [x for x in all_cols if x not in dropcolumns]
scaler = StandardScaler()
df_stacked[norm_cols] = scaler.fit_transform(df_stacked[norm_cols])

#%% one hot encoding
df_stacked = pd.concat([df_stacked, pd.get_dummies(df_stacked['location'], prefix='loc_')], axis = 1)
df_stacked = pd.concat([df_stacked, pd.get_dummies(df_stacked['season'], prefix='season_')], axis = 1)

#%% train, validation, test split
# trin-val split
train_val_data = df_stacked[df_stacked['date'] < '2019-12-01']
X = train_val_data[train_val_data.columns[~train_val_data.columns.isin(dropcolumns)]]
y = train_val_data['target_value'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

# test
test_data = df_stacked[(df_stacked['date'] >= '2019-12-01') & (df_stacked['date'] < '2020-01-01')]
X_test = test_data[test_data.columns[~test_data.columns.isin(dropcolumns)]]
y_test = test_data['target_value'].values

# extract indexes for the output data set
train_indexes = list(X_train.index)
val_indexes = list(X_val.index)
test_indexes = list(X_test.index)
#%% Training
learning_rate = 1e-3

# Creating AN
model = Sequential() 
model.add(Dense(units = 128, input_dim = X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
adam = optimizers.adam(lr = learning_rate)
model.add(Dense(1, activation='linear'))
model.compile(#loss = 'sparse_categorical_crossentropy',
                   loss = 'mean_squared_error',
                   optimizer = adam)

# Embedding layer
earlyStop = EarlyStopping(monitor = "val_loss"
                          , verbose = 100
                          , mode = 'auto'
                          , patience = 500)
# make the results reproducable
seed(123)
history = model.fit(X_train, y_train
          , batch_size = 5000     
          , epochs = 5000
          , verbose = 1
          , validation_data = (X_val, y_val)
          , callbacks = [earlyStop]
          )
print(model.summary())

#%% plot training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()

#%% validation metrics

# predictions
y_train_hat = model.predict(X_train)
y_val_hat = model.predict(X_val)
y_test_hat = model.predict(X_test)

# MSE
print("Mean squared error : \n\nTraining set:", 
      round(mean_squared_error(y_train, y_train_hat), 2), 
      "\nValidation set: ", 
      round(mean_squared_error(y_val, y_val_hat), 2),
      "\nTest set: ", 
      round(mean_squared_error(y_test, y_test_hat), 2))

# MAE
print("\n\n\nMean absolute error : \n\nTraining set:", 
      round(mean_absolute_error(y_train, y_train_hat), 2), 
      "\nValidation set: ", 
      round(mean_absolute_error(y_val, y_val_hat), 2),
      "\nTest set: ", 
      round(mean_absolute_error(y_test, y_test_hat), 2))

# MSE per group of days ahead in the test data
df_stacked[norm_cols] = scaler.inverse_transform(df_stacked[norm_cols])
test_data = df_stacked.iloc[test_indexes]
test_data['sq_error'] = (test_data['target_value'] - y_test_hat.ravel()) ** 2
print(test_data.groupby('days_ahead')['sq_error'].mean())

#%% save predictions in the required output form
df_stacked['days_ahead'] = 'Forecast_day_' + df_stacked['days_ahead'].astype(int).astype(str)

def output_df(ds_indexes, y_hat):
    """
    Returns a dataset (train/val/test) in a dataframe form, required by the task.
    
    ds_indexes: indexes of the coresponding kind of dataset, that should be extracted from he df_stacked.
    y_hat: predictions for the coresponding kind of dataset
    
    """
    
    ds = df_stacked.iloc[ds_indexes]
    ds['y_hat'] = y_hat
    ds = ds[['location', 'date', 'days_ahead', 'y_hat']]
    result = ds.pivot_table(index=['location', 'date'], 
                        columns='days_ahead', 
                        values='y_hat').reset_index()
    return result
train_result = output_df(ds_indexes = train_indexes, y_hat = y_train_hat)
train_result.to_csv("train_predictions.csv")
val_result = output_df(ds_indexes = val_indexes, y_hat = y_val_hat)
val_result.to_csv("val_predictions.csv")
test_result = output_df(ds_indexes = test_indexes, y_hat = y_test_hat)
test_result.to_csv("test_predictions.csv")

#%% save the model
timestamp = datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d__%H_%M_%S')
model.save("model_" + timestamp + ".h5")
print("Saved model to disk")
