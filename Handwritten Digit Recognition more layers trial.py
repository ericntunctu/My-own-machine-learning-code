#!/usr/bin/env python
# coding: utf-8

# In[14]:


from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential 
from keras.layers.core import Dense, Activation
from keras.layers import Dropout
from keras.utils import np_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


# In[3]:


## each data is a 2d object
X1_train = X_train.reshape(60000, 784)     
X1_test = X_test.reshape(10000, 784)

classes = 10
Y1_train = np_utils.to_categorical(Y_train, classes)     
Y1_test = np_utils.to_categorical(Y_test, classes)


input_size = 784
batch_size = 100     
hidden_neurons = 100    


# In[ ]:


### training process
model = Sequential()      # ˙784-1000-1000-10

model.add(Dense(units=1000, input_dim=784)) # Add Input/ first hidden_neurons 
model.add(Activation('sigmoid'))   
model.add(Dropout(0.5))  # Add DropOut functionality  

model.add(Dense(units=100, input_dim=100)) # Add Input/ second hidden_neurons 
model.add(Activation('sigmoid'))     
model.add(Dropout(0.5))  # Add DropOut functionality  

model.add(Dense(10, input_dim=100)) # Add second Input/ final results 
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')

model.fit(X1_train, Y1_train, batch_size=100, epochs=30, verbose=1)


# In[10]:


score = model.evaluate(X1_test, Y1_test, verbose=1) #算分數
print('Test accuracy:', score[1]) 


# In[ ]:


from keras.models import load_model

model.save('my model.h')

