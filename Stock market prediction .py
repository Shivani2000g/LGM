#!/usr/bin/env python
# coding: utf-8

# ## Lets Grow More LGM VIP Internship August(2022)
# ### Task-2:Stock Market Prediction and Forecasting using Stacked LSTM
# ### Author: Gore Shivani Kailas

# ### Importing the Library

# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

#ignoring warnings
import warnings
warnings.filterwarnings("ignore")


# ### Reading the Data

# In[38]:


df= pd.read_csv('C:/Users/Hp/OneDrive/Desktop/stock price.csv')
df.head()


# In[39]:


df.tail()


# ### Exploring Data and Making required change for better understanding

# In[40]:


#sort with date
df['Date'] = pd.to_datetime(df['Date'])
print(type(df.Date[0]))


# In[41]:


data=df.sort_values('Date')
data.head()


# In[42]:


data.reset_index(inplace=True)


# In[43]:


data.head()


# ### Visualizing the Data

# In[44]:


plt.plot(data['Close'])


# In[45]:


data1=data['Close']


# ### 1.Prepare Data

# In[46]:


## LSTM are sensitive to the scale of the data, therefore applying MinMax scaler
scaler=MinMaxScaler(feature_range=(0,1))
data1=scaler.fit_transform(np.array(data1).reshape(-1,1))
data1


# In[47]:


##splitting dataset into train and test split
training_size=int(len(data1)*0.70)
test_size=len(data1)-training_size
train_data,test_data=data1[0:training_size,:],data1[training_size:len(data1),:1]


# In[48]:


training_size,test_size


# In[5]:


#convert an array of values into a dataset matrix
def create_dataset(dataset,time_step=1):
    dataX, dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a= dataset[i:(i+time_step),0] ###i=0, 0,1,2,3-----99  100
        dataX.append(a)
        dataY.append(dataset[i + time_step,0])
    return np.array(dataX), np.array(dataY)


# In[6]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[51]:


print(X_train.shape), print(y_train.shape)


# In[52]:


print(X_test.shape) , print(ytest.shape)


# In[53]:


# reshape input to be [samples, time steps,features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# ### 1. Model Building

# In[54]:


### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[55]:


model= Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


# In[56]:


model.fit(X_train,y_train,validation_split=0.1,epochs=60,batch_size=64,verbose=1)


# In[57]:


##  Lets do the prediction and check performance metrics
test_predict=model.predict(X_test)


# In[58]:


## Transform back to original form
test_predict1=scaler.inverse_transform(test_predict)


# In[59]:


test_predict1


# In[60]:


## Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(ytest,test_predict))

