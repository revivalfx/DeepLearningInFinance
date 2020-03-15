#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Convolution1D, MaxPooling1D, Flatten,  Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer

np.random.seed(7)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore") 

t_0 = '2011-09-18'
t_n = '2020-03-01'

# This is an hourly csv file. Read it in and resample it.
eudf = pd.read_csv('/media/sf_tickdata/EURUSD-H1-2004.csv', parse_dates=True)
eudf['Date'] = pd.to_datetime(eudf['Date'],yearfirst=True, format='%Y%m%d')
#eudf['Date'] = pd.to_datetime(eudf['Date'])
eudf['Timestamp'] = eudf['Date'].astype(str) + ' ' + eudf['Timestamp'].astype(str)
eudf.index = pd.to_datetime(eudf['Timestamp'])
eudf = eudf.loc[eudf.index >= pd.to_datetime(t_0)]
eudf = eudf.loc[eudf.index <= pd.to_datetime(t_n)]
eudf = eudf[['Timestamp', 'Close']]
eudf.head()

#Create a daily data frame from an hourly
#eudf1 = eudf
eudf1 = eudf['Close'].astype('float').resample('D').asfreq()
eudf1.columns = ['Close']
print("eudf:\n", eudf1.head())

# S&P
spdf = pd.read_csv('/media/sf_tickdata/USA500IDXUSD.csv', index_col='Timestamp', parse_dates=True)
spdf['Date'] = pd.to_datetime(spdf['Date'],yearfirst=True, format='%Y%m%d')
spdf.index = spdf['Date']
spdf = spdf.loc[spdf.index >= pd.to_datetime(t_0)]
spdf = spdf.loc[spdf.index <= pd.to_datetime(t_n)]
spdf.tail()

#YEN dailies
ujdf = pd.read_csv('/media/sf_tickdata/USDJPY-2003-daily.csv', index_col='Timestamp', parse_dates=True)
ujdf['Date'] = pd.to_datetime(ujdf['Date'],yearfirst=True, format='%Y%m%d')
ujdf.index = ujdf['Date']
ujdf = ujdf.loc[ujdf.index >= pd.to_datetime(t_0)]
ujdf = ujdf.loc[ujdf.index <= pd.to_datetime(t_n)]

#Guppie Dailies
gjdf = pd.read_csv('/media/sf_tickdata/GBPJPY-2008-daily.csv', index_col='Timestamp', parse_dates=True)
gjdf['Date'] = pd.to_datetime(gjdf['Date'],yearfirst=True, format='%Y%m%d')
gjdf.index = gjdf['Date']
gjdf = gjdf.loc[gjdf.index >= pd.to_datetime(t_0)]
gjdf = gjdf.loc[gjdf.index <= pd.to_datetime(t_n)]

#Cable Dailies
gudf3 = pd.read_hdf('/media/sf_tickdata/GBPUSD-2008-daily.h5')
gudf3['Date'] = pd.to_datetime(gudf3['Date'],yearfirst=True, format='%Y%m%d')
gudf3.index = gudf3['Date']
gudf3 = gudf3.loc[gudf3.index > pd.to_datetime(t_0)]
gudf3 = gudf3.loc[gudf3.index <= pd.to_datetime(t_n)]

#plt.plot(gudf3.loc[gudf3.index > pd.to_datetime(t_0)]['Close'])
#plt.show()

#Create a historically cross-sectional financial context
#Start with S&P
fulldf = spdf
fulldf['S&P500']  = fulldf['Close']
fulldf = fulldf[['Date', 'S&P500']]

#Add Guppie
fulldf['GBPJPY'] = gjdf['Close']
#Add Yen
fulldf['USDJPY'] = ujdf['Close']
#Add Euro
fulldf['EURUSD'] = eudf1#['Close']
#Add Cable
fulldf['GBPUSD'] = gudf3['Close']
fulldf = fulldf.iloc[50:]


#plt.plot(gjdf['Close'])
#plt.plot(fulldf['GBPJPY'])
#plt.show()

#plt.plot(ujdf['Close'])
#plt.plot(fulldf['USDJPY'])
#plt.show()

#plt.plot(eudf1)#['Close'])
#plt.plot(fulldf['EURUSD'])
#plt.show()

#plt.plot(gudf3['Close'])
#plt.plot(fulldf['GBPUSD'])
#plt.show()
'''
prices  = pd.read_hdf('/media/sf_tickdata/fulldf.h5')

prices = prices.loc[prices.index > pd.to_datetime('2011-09-18')]

plt.plot(gudf3['Close'], "g")

plt.plot(gudf3.loc[gudf3.index > pd.to_datetime('2011-09-18')]['Close'], "g")
'''

def retFFT(key, df):
	n = len(df[key])
	Y = np.fft.fft(df[key])
	np.put(Y, range(100, n), 0.0)
	ifft = np.fft.ifft(Y)
	ifftdf = pd.DataFrame(ifft.real, index= df.index, columns=['Fourier Series'])
	#plt.plot(ifftdf[5:n-5], "r-")
	#plt.plot(df[key], "g")
	
	return ifftdf


prices = fulldf
prices['GBPUSD-ifft'] = retFFT('GBPUSD', prices)
prices['GBPJPY-ifft'] = retFFT('GBPJPY', prices)
prices['USDJPY-ifft'] = retFFT('USDJPY', prices)

print("prices:\n", prices.head())
print(prices.columns)
'''
n = len(prices['GBPUSD'])
Y = np.fft.fft(prices['GBPUSD'])
np.put(Y, range(100, n), 0.0)
ifft = np.fft.ifft(Y)

ifftdf = pd.DataFrame(ifft, index= prices.index, columns=['Fourier Series'])
'''

#plt.plot(gudf3.loc[gudf3.index > pd.to_datetime(t_0)]['Close'])


'''


# In[20]:


df = pd.read_hdf('../data/DeepLearning.h5', 'Data_GBPJPY')
#df = df.loc[df.index > pd.to_datetime('2008-01-01')]
df.head()


# In[121]:


didf = pd.read_hdf('../data/DeepLearning.h5', 'Data_Index')
#df = df.loc[df.index > pd.to_datetime('2008-01-01')]
didf.tail()


# In[122]:


dpdf = pd.read_hdf('../data/DeepLearning.h5', 'Deep_Portfolio')
#df = df.loc[df.index > pd.to_datetime('2008-01-01')]
dpdf.tail()


# In[119]:
import h5py

with h5py.File("../data/DeepLearning.h5") as f:
    print(f.keys())  




# In[6]:

'''

df = prices
del df['Date']

#prices[''].pct_change().fillna(0)
for c in df.columns:
    print("column: ", c)
    df[c+'_ret'] = df[c].pct_change().fillna(0)

def create_dataset(dataset, look_back=1, columns = ['GBPJPY']):
    dataX, dataY = [], []
    for i in range(len(dataset.index)):
        if i <= look_back:
            continue
        a = None
        for c in columns:
            b = dataset.loc[dataset.index[i-look_back:i], c].as_matrix()
            if a is None:
                a = b
            else:
                a = np.append(a,b)
        dataX.append(a)
        dataY.append(dataset.loc[dataset.index[i], columns].as_matrix())
    return np.array(dataX), np.array(dataY)


look_back = 12
sc = StandardScaler()
df.loc[:, 'GBPJPY'] = sc.fit_transform(np.array(df.loc[:, 'GBPJPY']).reshape(-1,1))
sc1 = StandardScaler()
df.loc[:, 'S&P500'] = sc1.fit_transform(np.array(df.loc[:, 'S&P500']).reshape(-1,1))
sc2 = StandardScaler()
df.loc[:, 'USDJPY'] = sc1.fit_transform(np.array(df.loc[:, 'USDJPY']).reshape(-1,1))
sc2 = StandardScaler()
df.loc[:, 'GBPUSD'] = sc1.fit_transform(np.array(df.loc[:, 'GBPUSD']).reshape(-1,1))


df.loc[:, 'GBPJPY-ifft'] = sc.fit_transform(np.array(df.loc[:, 'GBPJPY-ifft']).reshape(-1,1))
sc1 = StandardScaler()
df.loc[:, 'USDJPY-ifft'] = sc1.fit_transform(np.array(df.loc[:, 'USDJPY-ifft']).reshape(-1,1))
sc2 = StandardScaler()
df.loc[:, 'GBPUSD-ifft'] = sc1.fit_transform(np.array(df.loc[:, 'GBPUSD-ifft']).reshape(-1,1))

df.head()


train_df = df.loc[df.index < pd.to_datetime('2016-01-01')]
#val_df = train_df.loc[train_df.index >= pd.to_datetime('2013-01-01')]
#train_df = train_df.loc[train_df.index < pd.to_datetime('2013-01-01')]
#test_df = df.loc[df.index >= pd.to_datetime('2016-01-01')]
#train_x, train_y = create_dataset(train_df, look_back=look_back)
##val_x, val_y = create_dataset(val_df, look_back=look_back)
#test_x, test_y = create_dataset(test_df, look_back=look_back)
timeseries = np.asarray(df.GBPJPY)
timeseries = np.atleast_2d(timeseries)
if timeseries.shape[0] == 1:
        timeseries = timeseries.T
X = np.atleast_3d(np.array([timeseries[start:start + look_back] for start in range(0, timeseries.shape[0] - look_back)]))
y = timeseries[look_back:]

predictors = ['GBPJPY']#, 'GBPUSD','S&P500']#, 'USDJPY']
#TRAIN_SIZE = train_x.shape[0]
#EMB_SIZE = look_back
model = Sequential()
#model.add(Embedding(TRAIN_SIZE, 1, input_length=EMB_SIZE))
model.add(Convolution1D(input_shape = (look_back,1), 
                        nb_filter=64,
                        filter_length=2,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=2))

model.add(Convolution1D(input_shape = (look_back,1), 
                        nb_filter=64,
                        filter_length=2,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=2))

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(250))
model.add(Dropout(0.25))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss="mse", optimizer="rmsprop")
model.fit(X, 
          y, 
          epochs=1000, 
          batch_size=80, verbose=1, shuffle=False)



df['Pred'] = df.loc[df.index[0], 'GBPJPY']
for i in range(len(df.index)):
    if i <= look_back:
        continue
    a = None
    for c in predictors:
        b = df.loc[df.index[i-look_back:i], c].as_matrix()
        if a is None:
            a = b
        else:
            a = np.append(a,b)
        a = a
    y = model.predict(a.reshape(1,look_back*len(predictors),1))
    df.loc[df.index[i], 'Pred']=y[0][0]

df.to_hdf('DeepLearning.h5', 'Pred_CNN')
df.loc[:, 'GBPJPY'] = sc.inverse_transform(df.loc[:, 'GBPJPY'])
df.loc[:, 'Pred'] = sc.inverse_transform(df.loc[:, 'Pred'])
plt.plot(df.GBPJPY,'y')
plt.plot(df.Pred, 'g')
plt.show()

