#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 11:22:05 2023

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
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import pearsonr
import os

#from scipy.stats import norm
#from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
#from arch import arch_model

#%%

cwd = os.chdir('/Users/emildamirov/Desktop/Thesis/Analysis')

# Creating and Cleaning DataFrame 

df = pd.read_excel('Thesis Data.xlsx')
df = df[df['Date'] > '2010-01-03']
df.isnull().sum()

#%%

df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df = df.sort_values(by='Date', ascending = True)
df = df.drop(columns = ['MODEC1','MOA','MOZ23', 'ICEDEU3']).fillna(method='ffill')
df.index = df.index.date
df['lag_MO1'] = df['MO1'].shift(1)
df = df.dropna()
df.isnull().sum()

#%%
# Exploratory Data Analysis

# Plotting time-series 

#plot1 = df.plt.lineplot( y=['OVX', 'NEX', 'SP GSCI ', 'TZT1', 'XA1', 'CO1', 'MO1'], kind="line", figsize=(10, 8))
#fig = plot1.get_figure()
#fig.savefig(fname='Time-series')

fig, ax = plt.subplots(figsize=(8, 6))
plot = sns.lineplot(data=df[['MO1','OVX','NEX','SP GSCI ','TZT1','XA1','CO1']], 
             dashes=False, ax=ax)
plot.set(ylabel='EUR', title = 'Multivariate Time Series')
sns.move_legend(plot, "upper left")
plt.savefig('filename.png', dpi=300)


variables = ['OVX', 'NEX', 'SP GSCI ', 'TZT1', 'XA1', 'CO1', 'MO1']
for i in variables:
    df.plot( y=i, kind="line", figsize=(10, 8), title = i)
    
#%%
#1. Descriptive statistics
desc_stats = df.describe().transpose()
desc_stats = desc_stats.drop(['count','25%', '50%', '75%'], axis=1)

#2. Correlation table with ACF, PACF
# Estimate correlation of features: Pearson’s correlation

ll = []
for i in variables:
    corr, _ = pearsonr(df[i], df['MO1'])
    l = [i, corr]
    ll.append(l)
    
pear_corr = pd.DataFrame(ll, columns = ['Variable', 'Pearsons Correlation'])
pear_corr = pear_corr.set_index('Variable')
pear_corr.to_excel('output.xlsx')

# Distribution not relevant 
#plt.figure(figsize = (10,6))
#sns.distplot(df['MO1'])

#%%
# Dividing the dataset into training set and testing set

X = df[['lag_MO1','OVX','NEX','SP GSCI ','TZT1','XA1','CO1']].values
y = df['MO1'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4, random_state = 42)
X_train.shape

# Normalising the data

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test) 
early_stopping = EarlyStopping()

#%%
# Neural Network Model    
                                           # *check Sequential parameters
model = Sequential()
model.add(Dense(7, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))
model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mse')  # try changing the loss function; other optimizers?
model.fit(x = X_train, y = y_train, 
          validation_data = (X_test, y_test), epochs = 400)


#%%
loss_df = pd.DataFrame(model.history.history)
loss_df.plot(title = "Training Loss per Epoch")


fig, ax = plt.subplots(figsize=(7, 5))
plot = sns.lineplot(data=loss_df, 
             dashes=False, ax=ax)
plot.set(xlabel = 'Number of Epochs', ylabel='EUR', title = 'Training Loss per Epoch')
sns.move_legend(plot, "upper right")
plt.savefig('validation.png', dpi=300)


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
test_predictions = pd.Series(test_predictions.reshape(1345,))
pred_df = pd.concat([pred_df,test_predictions],axis=1)
pred_df.columns = ['Test Y','Model Predictions']

fig, ax = plt.subplots(figsize=(7, 5))
plot = sns.scatterplot(x='Test Y',y='Model Predictions',data=pred_df)
plot.set(xlabel = 'Actual Price', ylabel='Model Predictions')
plt.savefig('qualitative evaluation.png', dpi=300)


# Error distribution 
pred_df['Error'] = pred_df['Test Y'] - pred_df['Model Predictions']

fig, ax = plt.subplots(figsize=(7, 5))
sns.distplot(pred_df['Error'],bins=50)
plot.set(xlabel = 'Error', ylabel='Density', title = 'Error Distribution')
plt.savefig('error distribution.png', dpi=300)


#%%
# Quantitative evaluation
MAPE = mean_absolute_percentage_error(pred_df['Test Y'],pred_df['Model Predictions'])
MAE = mean_absolute_error(pred_df['Test Y'],pred_df['Model Predictions'])
MSE = mean_squared_error(pred_df['Test Y'],pred_df['Model Predictions'])
RMSE = test_score**0.5
EVS = explained_variance_score(pred_df['Test Y'],pred_df['Model Predictions'])
print(MAPE, MAE, MSE, RMSE, EVS)


#%%
# Saving the model
model.save('model_1.h5')

#%%

         




