# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 18:28:32 2021

@author: pawan
"""

import numpy as np
import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16
# Load model
model = VGG16()
# Summerize the model

# The model can then be used directly to classify a photograph into 1000 classes
model.summary()

from keras.preprocessing.image import load_img
from keras.applications.imagenet_utils import preprocess_input
# Laod an image from the file
image = load_img('mobile.jpg',target_size=(224,224))

# Convert the image pixels to numpy array
image = np.array(image)

# Resize the image for the model
image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))

# Prepare the image for VGG model
image = preprocess_input(image)
image

my_image = plt.imread('mobile.jpg')
plt.imshow(my_image)

# Predict the probabilty across all the classes
yhat = model.predict(image)
# yhat

# Convert the probabilties into classes
from keras.applications.vgg16 import preprocess_input,decode_predictions
label = decode_predictions(yhat)
label    

#####################################

import keras

from keras.models import Sequential
# Sequential from keras.models,This get our NN as sequential network
# As we know, it can be sequential or graph

from keras.layers import Dense,Activation,Dropout,Flatten,Conv2D, MaxPooling2D
#Importing Dense, Activation, Flatten,Activation,Dropout

from keras.layers.normalization import BatchNormalization
# For normalization 

# def VGG16_UP(input_tensor = None,classes=2):
    
#     img_rows,img_cols=300,300 # By default the size is 224,224
#     img_channels = 3
    
#     img_dim = (img_rows,img_cols,img_channels)
    
#     img_input = Input(shape=img_dim)
    
#     # Block1
#     x= Conv2D()


# Test1
l=[1,2,3,4,5] # K=2
#Test2
l=[7,9,6,6,7,8,3,0,9,5]  # k=5
# Test 3  # k=1
l = [1]
# Test 4  # K=1
l = [1,2]

# Test 5 k = 2
l = [1,2,3]

k=2
input_value_indx=k-1
value_at_index = l[input_value_indx]
from_end_indx = l[-(k)]
#replace_at_begin
l[input_value_indx] = l[-(k)]
l[-(k)]= value_at_index


def swapNodes(l,k):
    input_value_indx=k-1
    value_at_index = l[input_value_indx]
    from_end_indx = l[-(k)]
    #replace_at_begin
    l[input_value_indx] = l[-(k)]
    l[-(k)]= value_at_index
    return l
    
ListNode = [1,2,3,4,5,6,7]
class Solution:
    def swapNodes(self, head: ListNode, k: int) -> ListNode:
        A, B = head, head
        for i in range(1, k): A = A.next
        nodeK, A = A, A.next
        while A: A, B = A.next, B.next
        nodeK.val, B.val = B.val, nodeK.val
        return head


    
  
    
    
    
a_list = ["a", "b", "c"]
index1 = a_list. index("a")
index2 = a_list. index("c")
a_list[index1], a_list[index2] = a_list[index2], a_list[index1]
print(a_list)
    
