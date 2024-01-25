# -*- coding: utf-8 -*-
"""emotion_recognition_ferplus(drop contemp/unknown/NF).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aloNlEVJrpfIjd9KQRkQBaMAvIXVTRYr
"""

from google.colab import drive
drive.mount('/content/drive')

import numpy as np 
import pandas as pd 



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Input, BatchNormalization, MaxPooling2D, Activation, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from keras.preprocessing import image
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau

import scipy
import os
import cv2

data = pd.read_csv('/content/drive/MyDrive/專題/1_new/fer2013new1.csv')

data = data.drop(columns=['contempt', 'unknown','NF'])

data

emotion = ['neutral','happiness','surprise','sadness','anger','disgust','fear']
#數字替代
emo=data[emotion].idxmax(axis=1).replace('neutral', 0)
emo = emo.replace('happiness', 1)
emo = emo.replace('surprise', 2)
emo = emo.replace('sadness', 3)
emo = emo.replace('anger', 4)
emo = emo.replace('disgust', 5)
emo = emo.replace('fear', 6)
# emo = emo.replace('contempt', 8)
# emo = emo.replace('unknown', 9)
# emo = emo.replace('NF', 10)

# emo=data[emo].idxmax(axis=1).replace('happiness', 0)
# emo=data[emo].idxmax(axis=1).replace('surprise', 0)
# emo=data[emo].idxmax(axis=1).replace('sadness', 0)
# emo=data[emo].idxmax(axis=1).replace('anger', 0)
# emo=data[emo].idxmax(axis=1).replace('disgust', 0)
# emo=data[emo].idxmax(axis=1).replace('fear', 0)
# emo=data[emo].idxmax(axis=1).replace('contempt', 0)
# emo=data[emotion].idxmax(axis=1).replace('unknown', 0)
# emo=data[emotion].idxmax(axis=1).replace('NF', 0)
# print(emo)
emo

x_data = data['pixels']
y_data = emo

y_data.value_counts() #neutral最多

#視覺化看數量 compare
sns.set_theme(style="darkgrid")
ax = sns.countplot(x=emo, data=data)

#進行上採樣
oversampler = RandomOverSampler(sampling_strategy='auto')
x_data, y_data = oversampler.fit_resample(x_data.values.reshape(-1,1), y_data) #數據變成一行
print(x_data.shape," ",y_data.shape)

y_data.value_counts() #取數量最多的情緒張數(補成一樣數量)

x_data = pd.Series(x_data.flatten()) #把高維度資訊拉成直線
x_data

x_data = np.array(list(map(str.split, x_data)), np.float32) #資料型態轉換
x_data/=255 #圖像歸一化
x_data[:10]

x_data = x_data.reshape(-1, 48, 48, 1)
x_data.shape

y_data = np.array(y_data)
y_data = y_data.reshape(y_data.shape[0], 1)
y_data.shape

#early stopping but useless :D
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.00005,
    patience=8,
    verbose=1,
    restore_best_weights=True,
)
callbacks = [
    early_stopping
    ]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.1, random_state = 45) #測試訓練1:9

model = Sequential([
    Input((48, 48, 1)),
    Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='valid'),
    BatchNormalization(axis=3),
    Activation('relu'), #Rectified Linear Activation 在神經網路中表現佳
    Conv2D(64, (3,3), strides=(1,1), padding = 'same'),
    BatchNormalization(axis=3),
    Activation('relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), strides=(1,1), padding = 'valid'),
    BatchNormalization(axis=3),
    Activation('relu'),
    Conv2D(128, (3,3), strides=(1,1), padding = 'same'),
    BatchNormalization(axis=3),
    Activation('relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), strides=(1,1), padding = 'valid'),
    BatchNormalization(axis=3),
    Activation('relu'),
    MaxPooling2D((2,2)),
    Flatten(),#連結Conv2ED與Dense
    Dense(200, activation='relu'),
    Dropout(0.6), #Drop掉一定比例的神經元來避免Overfit
    Dense(7, activation = 'softmax')
])
model.summary()

adam = keras.optimizers.Adam(learning_rate=0.0001) #學習率0.0001
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

y_train = np_utils.to_categorical(y_train, 7)
y_train.shape

y_test = np_utils.to_categorical(y_test, 7)
y_test.shape

history = model.fit(x_train, y_train, epochs = 35, validation_data=(x_test, y_test),callbacks=callbacks)

print("Accuracy of our model on validation data : " , model.evaluate(x_test,y_test)[1]*100 , "%")

print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

results = model.evaluate(x_train, y_train, batch_size=128)
print("train loss, train acc:", results)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#plot show
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

y_pred = model.predict(x_test)
y_result = []

for pred in y_pred:
    y_result.append(np.argmax(pred))
y_result[:10]

y_actual = []

for pred in y_test:
    y_actual.append(np.argmax(pred))
y_actual[:10]

from sklearn.metrics import confusion_matrix, classification_report #各情緒準確率
print(classification_report(y_actual, y_result))

#熱圖
import seaborn as sn
cm = tf.math.confusion_matrix(labels = y_actual, predictions = y_result)

plt.figure(figsize = (10, 7))
sn.heatmap(cm, annot = True, fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Truth')

#儲存模型
fer_json = model.to_json()  
with open("fer.json", "w") as json_file:  
    json_file.write(fer_json)  
model.save("/content/drive/MyDrive/專題/1_new/mode35new_es10.h5")