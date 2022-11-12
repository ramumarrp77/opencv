import requests
from bs4 import BeautifulSoup
import lxml
import os
import urllib
import sys
import pandas as pd
import numpy as np
from PIL import Image
import gc
import cv2
import csv
import matplotlib.pyplot as plt
###############################################################################################
#https://bulkresizephotos.com/zh-tw <- This website can change your image to 32*32 pixels
new_train=pd.read_csv('new_train_id0_499.csv')
filename=os.listdir("BulkResizePhotos")
filename.sort(key=lambda x:float(x[:-4]))#Watch out!You have to read data from 0 to 27776 in order
img=[]
for file in filename:
	img.append(np.array(Image.open("BulkResizePhotos\\"+file)))
img=np.array(img)
###########################################################################################
np.random.seed(1337)
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D,Dense,MaxPool2D,Activation,Dropout,Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(img,new_train['landmark_id'],test_size=0.2)
y_train=y_train.astype(int)
y_test=y_test.astype(int)
y_train=np.array(y_train).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)
X_train=X_train.reshape(-1,32,32,3)/255 #Normalize
X_test=X_test.reshape(-1,32,32,3)/255
y_train=np_utils.to_categorical(y_train,num_classes=500)
y_test=np_utils.to_categorical(y_test,num_classes=500)#landmark_id is from 0 to 499
########################################################################################
model=Sequential()
model.add(Convolution2D(filters=32,kernel_size=(3,3),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(Dropout(rate=0.4))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))

model.add(Convolution2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(Dropout(rate=0.4))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))

model.add(Flatten())

model.add(Dense(1024,activation='relu'))
model.add(Dropout(rate=0.4))

model.add(Dense(500,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

train_history=model.fit(X_train,y_train,validation_split=0.2,epochs=40,batch_size=128,verbose=1)
accuracy=model.evaluate(X_test,y_test,verbose=1)
print(accuracy[1])



def show_train_history(train_history,train,validation):
	plt.plot(train_history.history[train])
	plt.plot(train_history.history[validation])
	plt.title('Train History')
	plt.ylabel('train')
	plt.xlabel('Epoch')
	plt.legend(['train','validation'],loc='upper left')
	plt.show()

show_train_history(train_history,'accuracy','val_accuracy') #acc:accuracy for training set . val_acc:accuracy for validation 
