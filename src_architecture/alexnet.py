# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:12:34 2021

@author: pawan
"""


import keras

from keras.models import Sequential
# Sequential from keras.models,This get our NN as sequential network
# As we know, it can be sequential or graph

from keras.layers import Dense,Activation,Dropout,Flatten,Conv2D, MaxPooling2D
#Importing Dense, Activation, Flatten,Activation,Dropout

from keras.layers.normalization import BatchNormalization
# For normalization 

import numpy as np

image_shape = (227,227,3)

np.random.seed(1000)
# Instantiate an empty model

model = Sequential()
# It start here

#1st convoluation layer
model.add(Conv2D(filters=96,input_shape = image_shape,kernel_size = (11,11),strides = (4,4),padding = 'valid'))
model.add(Activation("relu"))
# First layer has 96 Filers ,the input shape is 227x227x3
# Kernel size 11x11 striding 4x4 , Relu is the activation function

# Max Pooling Layer
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding ='valid'))

# 2nd Convolution layer
model.add(Conv2D(filters=256,input_shape=image_shape,kernel_size=(5,5),strides=(1,1),padding='valid'))
model.add(Activation("relu"))
#Max Pooling Layer
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid'))

# 3rd Convolution Layer
model.add(Conv2D(filters=384,input_shape=image_shape,kernel_size=(3,3),strides =(1,1),padding='valid'))
model.add(Activation('relu'))

# 4th Convolution Layer
model.add(Conv2D(filters=384,input_shape=image_shape,kernel_size=(3,3),strides =(1,1),padding='valid'))
model.add(Activation('relu'))

# 5th Convolution Layer
model.add(Conv2D(filters=256,input_shape=image_shape,kernel_size=(3,3),strides =(1,1),padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid'))

# Passing it to the fully connected layer ,Here we do the flatten!
model.add(Flatten())

#1st Fully Connected Layer has 4096 neurons
model.add(Dense(4096,input_shape=(227,227,3)))
model.add(Activation('relu'))
# Add Dropout to prrevent overfitting
model.add(Dropout(0.4))

# 2nd fully connected layer
model.add(Dense(4096))
model.add(Activation('relu'))
#Add Dropout
model.add(Dropout(0.4))

# Output layer
model.add(Dense(1000))
model.add(Activation('softmax'))

model.summary()

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=["accuracy"])










