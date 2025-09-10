# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 20:03:03 2019

@author: Johannes
"""

#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/


# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
file ="C:\\Users\\Johannes\\Programming\\python\\keras_example\\pima-indians-diabetes.data.csv"

dataset = loadtxt(file, delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

