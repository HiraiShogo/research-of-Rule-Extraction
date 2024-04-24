from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf

import keras

import sys, os

# 学習データをロード
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 正規化する
x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
# onehotベクトル表現にする
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape, type(x_train))
print(y_train.shape, type(y_train))
print(x_test.shape, type(x_test))
print(y_test.shape, type(y_test))


# モデルを定義
initial_model = Sequential(
    [
        Conv2D(filters=10, kernel_size=(3, 3),padding='same',activation='sigmoid',input_shape=(28,28,1),name='Conv'),
        MaxPooling2D(pool_size=(2, 2), strides=2, name='MaxPooling'),
        Flatten(name='Flatten'),
        Dense(10, activation='softmax', name="Dense")
    ]
)


initial_model.summary()

initial_model.compile(optimizer=RMSprop(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])

# モデルのトレーニングを実施
history = initial_model.fit(x=x_train, y=y_train,validation_data=(x_test,y_test), shuffle=True, epochs=10, batch_size=1000)
initial_model.save('model/CNN_1layer')

import matplotlib.pyplot as plt
import numpy as np

acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

# x軸としてepochsを用意
epochs = np.arange(1, len(acc) + 1, 1)

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, label='training_acc')
plt.plot(epochs, val_acc, label='validation_acc')
plt.xlabel('epochs')
plt.ylabel('accuracy [%]')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='training_loss')
plt.plot(epochs, val_loss, label='validation_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.legend()
plt.show()