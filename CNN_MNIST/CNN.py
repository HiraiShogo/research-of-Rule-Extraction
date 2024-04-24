from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf

import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import sys

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
initial_model = load_model('model/CNN_1layer')
initial_model.summary()
feature_extractor = keras.Model(inputs=initial_model.inputs,outputs=initial_model.get_layer(name="Flatten").output,)

x = x_train[5].reshape(1,28,28,1)
features = feature_extractor(x).numpy()[0]
print(features)
print(features.shape)


def img_show(img, i):
    pil_img = Image.fromarray(np.uint8(img))
    filename = 'pic/'+str(i)+'.png'
    pil_img.save(filename)


'''
for i in range(5,10):
    x = x_train[i].reshape(1,28,28,1)
    features = feature_extractor(x).numpy()[0]
    print(features)
    print(np.argmax(features))
    img_show(x_train[i].reshape(28, 28) * 255, i)
'''






