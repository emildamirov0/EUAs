#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 14:49:44 2023

@author: emildamirov
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error, explained_variance_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import *
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import InputLayer, LSTM
from scipy.stats import pearsonr

import os

#%%

cwd = os.chdir('/Users/emildamirov/Desktop/Thesis/Analysis')

df = pd.read_excel('Thesis Data.xlsx')
df = df[df['Date'] > '2010-01-03']
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df = df.sort_values(by='Date', ascending = True)
df = df.drop(columns = ['MODEC1','MOA','MOZ23', 'ICEDEU3', 'OVX']).fillna(method='ffill')
df.index = df.index.date
df['lag_MO1'] = df['MO1'].shift(1)
df = df.dropna()
df.isnull().sum()


#%%

X = df[['lag_MO1','NEX','SP GSCI ','TZT1','XA1','CO1']].values
num_samples, num_features = X.shape
y = df['MO1'].values
num_timesteps = 3
y = y[num_timesteps - 1:]

n = np.zeros((num_samples - num_timesteps + 1, num_timesteps, num_features))

for start_pos in range(num_timesteps):
    end_pos = start_pos + n.shape[0]
    n[:, start_pos, :] = X[start_pos:end_pos]

X_train, X_test, y_train, y_test = train_test_split(n,y, test_size = 0.4, random_state = 42)
X_train.shape
y_train.shape

#%%
                            #GRU?
# LSTM model

model = Sequential()
model.add(InputLayer((3,6)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(25, 'sigmoid'))
model.add(Dense(1, 'linear'))
model.summary()

model.compile(optimizer = 'adam', loss = 'mse')
model.fit(x = X_train, y = y_train, 
          validation_data = (X_test, y_test), epochs = 400)

model.save('model_1.h5')

#%%


loss_df = pd.DataFrame(model.history.history)
loss_df.plot(title = "Training Loss per Epoch")


fig, ax = plt.subplots(figsize=(7, 5))
plot = sns.lineplot(data=loss_df, 
             dashes=False, ax=ax)
plot.set(xlabel = 'Number of Epochs', ylabel='EUR', title = 'LSTM')
sns.move_legend(plot, "upper right")
plt.savefig('LSTM_validation.png', dpi=300)


#%%
# Evaluation
model.metrics_names
training_score = model.evaluate(X_train,y_train,verbose=0)
test_score = model.evaluate(X_test,y_test,verbose=0)
print(training_score,test_score)

#%%
# Further qualitative evaluation

# Plotting test/actual data and predictions from the model 
test_predictions = model.predict(X_test)
pred_df = pd.DataFrame(y_test,columns=['Test Y'])
test_predictions = pd.Series(test_predictions.reshape(1344,))
pred_df = pd.concat([pred_df,test_predictions],axis=1)
pred_df.columns = ['Test Y','Model Predictions']

fig, ax = plt.subplots(figsize=(7, 5))
plot = sns.scatterplot(x='Test Y',y='Model Predictions',data=pred_df)
plot.set(xlabel = 'Actual Price', ylabel='Model Rredictions')
plt.savefig('LSTM_qualitative_evaluation.png', dpi=300)
#%%

# Error distribution 
pred_df['Error'] = pred_df['Test Y'] - pred_df['Model Predictions']

fig, ax = plt.subplots(figsize=(7, 5))
sns.distplot(pred_df['Error'],bins=50)
plot.set(xlabel = 'Error', ylabel='Density', title = 'LSTM')
plt.savefig('LSTM_error_distribution.png', dpi=300)


#%%
# Quantitative evaluation
MAPE = mean_absolute_percentage_error(pred_df['Test Y'],pred_df['Model Predictions'])
MAE = mean_absolute_error(pred_df['Test Y'],pred_df['Model Predictions'])
MSE = mean_squared_error(pred_df['Test Y'],pred_df['Model Predictions'])
RMSE = test_score**0.5
EVS = explained_variance_score(pred_df['Test Y'],pred_df['Model Predictions'])
print(MAPE, MAE, MSE, RMSE, EVS)

