# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import keras

from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.cross_validation import  train_test_split
import time #helper libraries
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import newaxis

dataset=pd.read_csv("/Users/xuetan/Desktop/Ivey_Business Analytics/Programming/Rworkfile/bitcoin1.csv")
price=dataset['Close']
train=price[10:1840]
train=pd.DataFrame(train)
sc=MinMaxScaler()
train=sc.fit_transform(train)
X_train=train[0:len(train)-1]
Y_train=train[1:len(train)]
X_train=np.reshape(X_train,(len(X_train),1,1))

######build model
regressor=Sequential()
regressor.add(LSTM(units=1,activation='sigmoid',input_shape=(None,1)))
regressor.compile(optimizer='adam',loss='mean_squared_error')
regressor.fit(X_train,Y_train,batch_size=5,epochs=100)

####predict
test=price[1:10]
test=pd.DataFrame(test)
inputs=np.reshape(test,(len(test),1))
inputs=sc.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predict=regressor.predict(inputs)
predict=sc.inverse_transform(predict)


# Visualising the results
plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')
ax = plt.gca()  
plt.plot(test, color = 'red', label = 'Real BTC Price')
plt.plot(predict, color = 'blue', label = 'Predicted BTC Price')
plt.title('BTC Price Prediction', fontsize=40)
df_test=dataset[1:10]
df_test = df_test.reset_index()
x=df_test.index
labels = df_test['date']
plt.xticks(x, labels, rotation = 'vertical')
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
plt.xlabel('Time', fontsize=40)
plt.ylabel('BTC Price(USD)', fontsize=40)
plt.legend(loc=2, prop={'size': 25})
plt.show()