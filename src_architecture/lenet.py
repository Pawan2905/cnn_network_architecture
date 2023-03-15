# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt

import tensorflow as tf

import numpy as np

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

rows,cols =28,28

x_train = x_train.reshape(x_train.shape[0],rows,cols,1)
x_test = x_test.reshape(x_test.shape[0],rows,cols,1)

input_shape = (rows,cols,1)

# Normalize
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255.0
x_test = x_test/255.0

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)


def build_lenet(input_shape):
    
    # Sequential api
    model = tf.keras.Sequential()
    # Convulation1 : Filter as we know is 6,filter size is 5x5 , tanh is the activation function.28x28 dimension
    model.add(tf.keras.layers.Conv2D(filters=6,
                                     kernel_size=(5,5),
                                     strides = (1,1),
                                     activation='tanh',
                                     input_shape= input_shape))
    
    # Subsampling 1: Input =28x28.Output=14x14x6.Subsampling is simply Average Pooling so we use avg_pool
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2),
                                               strides=(2,2))
              
    # Convulation 2: Input 14x14x6 ,Output = 10x10x16 conv2d
    model.add(tf.keras.layers.Conv2D(filters=16,
                                     kernel_size=(5,5),
                                     strides = (1,1),
                                     activation='tanh'))
      
    # Subsampling 2 : Input=28x28x6 .Output = 14x14x6.Subsampling is simply Average pooling so we use avg_pool
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2),
    						 strides=2,2))
    
    model.add(tf.keras.layers.Flatten())
    # We must flatten the further steps to happen
    # It is the process of converting all the 2D layers as single long continuous linear vector
    
    model.add(tf.keras.layers.Dense(units=120,activation='tanh'))
    # Fully connected layer 1. Input size =5x5x16 output =120
    
    model.add(tf.keras.layers.Flatten())
    
    model.add(tf.keras.layers.Dense(units=84,activation='tanh'))
    # Fully connected layer # Input =120,Output =84
    
    # Output layer
    model.add(tf.keras.layers.Dense(units=10,activation='softmax'))
    # Final ,Output and activation through softmax
    
    model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(lr=0.1,momentum=0.0,decay=0.0,
                                                                                    metrics=['accuracy']))
    # Arguments passed aree like the past, nothing to worry!
    
    return model

lenet =build_lenet(input_shape)
# We build it!

# Number of epochs
epochs =10
# Can we train model?
history = lenet.fit(x_train,y_train,
                    epochs =epochs,
                    batch_size= 128,
                    verbose=1)  
loss,acc  = lenet.evaluate(x_test, y_test) 
print('Accuracy: 'acc)

# Transforming and reshape in 28x28 pixel
x_train = x_train.reshape(x_train.shape[0],28,28)
print('Training Data ',x_train.shape,y_train.shape)

x_test = x_test.reshape(x_test.shape[0],28,28)

# To visualize a single image at index 8888 (6 in the dataset)
# image_index =8888
# plt.imshow(x_test[image_index].reshape(28,28),cmap='Greys')

# To predict the output using the lenet model built 
# prep = lenet.predict(x_test[image_index].reshape(1,rows,cols,1))
# print(prep.argmax())

# Example 2 image@index 4444 (9 is the number in the dataset)
image_index=4444
plt.imshow(x_test[image_index].reshape(28,28),cmap='Greys')

pred = lenet.predict(x_test[image_index].reshape(1,rows,cols,1))        
