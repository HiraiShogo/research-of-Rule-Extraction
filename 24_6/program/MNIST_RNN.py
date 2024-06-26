import tensorflow as tf
from tensorflow.keras.datasets import mnist

import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F

import numpy as np
from PIL import Image

import os,sys
os.chdir(os.path.dirname(os.path.abspath((__file__))))


# 文字画像表示
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def mnist_import():
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    return (x_train, y_train),(x_test, y_test)

def load_mnist():
    x_train = np.array([])
    y_train = np.array([])
    x_test = np.array([])
    y_test = np.array([])

    for i in range(10):
        im_train = np.array(Image.open('dataset/MNIST/train/'+str(i)+'.png')).reshape(1,28,28)
        x_train = np.append(x_train, im_train).reshape(-1,28,28)
        y_train = np.append(y_train, i)

        im_test = np.array(Image.open('dataset/MNIST/test/' + str(i) + '.png')).reshape(1,28,28)
        x_test = np.append(x_test, im_test).reshape(-1,28,28)
        y_test = np.append(y_test, i)

    #形表示
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    return (x_train, y_train),(x_test, y_test)


(x_train, y_train),(x_test, y_test) = load_mnist()


# 学習済みモデルの読み込み
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()
model.to(device)

# 出力
y = model(x_test)
print(y)