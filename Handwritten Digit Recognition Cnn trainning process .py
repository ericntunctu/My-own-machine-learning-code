#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np     
np.random.seed(0)  #for reproducibility            

from keras.datasets import mnist
from keras.models import Sequential 
from keras.layers import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten

from keras.utils import np_utils

## machine learning parameters
input_size = 784
batch_size = 100     
hidden_neurons = 200
classes = 10     
epochs = 1

#load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#reshape the data
X_train = X_train.reshape(60000, 28, 28, 1)     
X_test = X_test.reshape(10000, 28, 28, 1)

X_train = X_train.astype('float32')     
X_test = X_test.astype('float32')     
X_train /= 255     
X_test /= 255

Y_train = np_utils.to_categorical(Y_train, classes)     
Y_test = np_utils.to_categorical(Y_test, classes)

model = Sequential() 

model.add(Convolution2D(32, (3, 3), input_shape=(28, 28, 1)))
###  ReLU stands for Rectified Linear Unit for a non-linear operation. The output is ƒ(x) = max(0,x).
model.add(Activation('relu'))

### ReLU stands for Rectified Linear Unit for a non-linear operation. The output is ƒ(x) = max(0,x).
model.add(Convolution2D(32, (3, 3)))  
model.add(Activation('relu'))
### Max pooling take the largest element from the rectified feature map. 
### Taking the largest element could also take the average pooling. 
### Sum of all elements in the feature map call as sum pooling.

model.add(MaxPooling2D(pool_size=(2, 2))) 
###prevent over fitting
model.add(Dropout(0.25))  
               

model.add(Flatten())
model.add(Dense(hidden_neurons)) 
model.add(Activation('relu'))      
model.add(Dense(classes)) 
model.add(Activation('softmax'))
     
### standard ## many details
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adadelta')

#start trainning
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split = 0.1, verbose=1)


## check the scores
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test accuracy:', score[1]) 


## save your model
from keras.models import load_model

model.save('convolution mnist.h')


# In[ ]:




