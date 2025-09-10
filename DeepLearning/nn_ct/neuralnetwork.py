# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 19:48:48 2019

@author: Johannes
"""

#%% Daten laden
from keras.datasets import mnist
train_da, test_da = mnist.load_data()
x_train, y_train = train_da
x_test, y_test = test_da




#%% Daten normalisieren
import keras.backend as K
from keras.utils import to_categorical

dat_from = K.image_data_format()
rows,cols = 28 ,28
train_size = x_train.shape[0]
test_size = x_test.shape[0]

if dat_from == 'channels_first':
    x_train = x_train.reshape(train_size,1,rows,cols)
    x_test = x_test.reshape(test_size,1,rows,cols)
    input_shape = (1,rows,cols)
else:
    x_train = x_train.reshape(train_size, rows,cols,1)
    x_test = x_test.reshape(test_size, rows,cols,1)
    input_shape = (rows,cols,1)

#norm data to float in tange 0..1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train/= 255
x_test/= 255
#conv class vecs to one hot vec
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)


#%%Daten kuerzen
x_train = x_train[:10]
y_train = y_train[:10]
    
    
#%%
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout, Conv2D, MaxPooling2D


model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(200,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

#%%
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
model.compile(loss=categorical_crossentropy,optimizer=Adam(),metrics=['accuracy'])

#%%training starten
history = model.fit(x_train,y_train,batch_size = 128, epochs=12, verbose=1,validation_data = (x_test,y_test))


#%%
from pandas import DataFrame
df_loss = DataFrame(data={
    'Epoche': history.epoch * 2,
    'Legende': ['Loss auf Trainingsdaten'] * len(history.epoch) + ['Loss auf Testdaten'] * len(history.epoch),
    'Loss': history.history['loss'] + history.history['val_loss']
})
df_accuracy = DataFrame(data={
    'Epoche': history.epoch * 2,
    'Legende': ['Accuracy auf Trainingsdaten'] * len(history.epoch) + ['Accuracy auf Testdaten'] * len(history.epoch),
    'Accuracy': history.history['accuracy'] + history.history['val_accuracy']
})

import altair as alt
chart_loss = alt.Chart(df_loss).mark_line().encode(
    x='Epoche', y='Loss', color='Legende')
chart_accuracy = alt.Chart(df_accuracy).mark_line().encode(
    x='Epoche', y='Accuracy', color='Legende')
chart = chart_loss + chart_accuracy
chart.resolve_scale(y='independent')
chart.save('chart.html')

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
