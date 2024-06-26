'''
Created on 2019/06/21

@author: ail
'''

import sys
import numpy as np
import chainer

from networks.LeNet_alt7 import LeNet_alt7 as DNN  # @UnresolvedImport

from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer.datasets import tuple_dataset
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import cupy as cp
from collections import OrderedDict
import wx
from PIL import Image
import threading
import matplotlib.pyplot as plt
import noise
import math

"""
class DNN(Chain):
    def __init__(self):
        super(DNN, self).__init__()
        self.output = {}
        with self.init_scope():
            #============================================================================
            #引数設定
            #
            #  L.Convolution2D((入力数), (出力数), (フィルタサイズ), stride =1, pad = 0)
            #  L.Linear((入力数), (出力数))
            #============================================================================

            self.conv1 = L.Convolution2D(1, 10, 5)
            self.conv2 = L.Convolution2D(10, 10, 8)
            self.conv3 = L.Convolution2D(10, 10, 16)
            self.affine1 = L.Linear(10, 10)
            self.affine2 = L.Linear(10, 10)

    def __call__(self, x):
        out = x
        self.output = {}
        #================================================================================
        #活性化関数設定
        #
        #  F.relu((入力))
        #  F.sigmoid((入力))
        #  F.max_pooling_2d((入力), (フィルタサイズ), stride = None, pad = 0)
        #  F.dropout((入力), ratio=.5)
        #================================================================================
        out = F.relu(self.conv1(out))
        self.output["Conv1"] = out.array
        out = F.relu(self.conv2(out))
        self.output["Conv2"] = out.array
        out = F.sigmoid(self.conv3(out))
        self.output["Conv3"] = out.array
        out = F.max_pooling_2d(out, 2)
        self.output["Pool"] = out.array
        out = F.sigmoid(self.affine1(out))
        self.output["Affine1"] = out.array
        self.output["Affine2"] = self.affine2(out).array
        return self.affine2(out)

    def culc(self, x):
        out = x
        self.output = {}
        #================================================================================
        #活性化関数設定
        #
        #  F.relu((入力))
        #  F.sigmoid((入力))
        #  F.max_pooling_2d((入力), (フィルタサイズ), stride = None, pad = 0)
        #  F.dropout((入力), ratio=.5)
        #================================================================================
        out = F.relu(self.conv1(out))
        self.output["Conv1"] = out.array
        out = F.relu(self.conv2(out))
        self.output["Conv2"] = out.array
        out = F.sigmoid(self.conv3(out))
        self.output["Conv3"] = out.array
        out = F.max_pooling_2d(out, 2)
        self.output["Pool"] = out.array
        out = F.sigmoid(self.affine1(out))
        self.output["Affine1"] = out.array
        self.output["Affine2"] = self.affine2(out).array
        return self.affine2(out)

    def get_output(self):
        return self.output
"""


class Model:
    def __init__(self):
        self.dnn = DNN()
        self.model = L.Classifier(self.dnn)
        self.params = None
        self.train, self.test = datasets.get_mnist(ndim=3)
        self.size = None
        self.d = 0.004
        self.sigma = 0.1
        self.output_part = ""
        self.path = ""

    def train_Network(self, n_in, n_units, n_out, gd=0, e=50, bs=100, f=-1):
        self.params = {"gd":gd, "e":e, "bs":bs, "f":f, "n_in":n_in, "n_units":n_units, "n_out":n_out}
        chainer.cuda.get_device_from_id(0)
        self.model.to_gpu()

        optimizer = chainer.optimizers.Adam()
        optimizer.setup(self.model)

        train_iter = chainer.iterators.SerialIterator(self.train, self.params["bs"])
        test_iter = chainer.iterators.SerialIterator(self.test, self.params["bs"], repeat=False, shuffle=False)

        updater = training.StandardUpdater(train_iter, optimizer, device=self.params["gd"])
        trainer = training.Trainer(updater, (self.params["e"], 'epoch'))

        trainer.extend(extensions.Evaluator(test_iter, self.model,device=self.params["gd"]))
        trainer.extend(extensions.dump_graph('main/loss'))

        frequency = self.params["e"] if self.params["f"] == -1 else max(1, self.params["f"])
        trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
        trainer.extend(extensions.LogReport())
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
                                  'epoch', file_name='accuracy.png'))
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.run()

    def save_Network(self):
        serializers.save_npz('my_mnist.npz', self.model)

    def load_Network(self,path =""):
        if path == "":
            app = wx.App()
            filter = "network file(*npz) | *.npz| All file(*.*) | *.*"
            dialog = wx.FileDialog(None, u'ネットワークファイルを選択してください', '', '', filter)
            dialog.ShowModal()
            path = dialog.GetPath()
        serializers.load_npz(path, self.model)
        chainer.cuda.get_device_from_id(0)
        self.model.to_gpu()

    def predict_Test(self, num):
        #print(np.array(self.test[0]).shape)
        prediction = self.model.predictor(cp.array(self.test[num][0][0]).reshape(1, 1, 28, 28))
        probability = chainer.cuda.to_cpu(F.softmax(prediction).data[0])
        np.set_printoptions(precision=20, floatmode='fixed', suppress=True)
        ans = self.dnn.get_output()
        ans["output"] = probability
        #print(probability)
        #print(self.dnn.get_output())
        return ans

    def predict_Train(self, num):
        #print(np.array(self.test[0]).shape)
        prediction = self.model.predictor(cp.array(self.train[num][0][0]).reshape(1, 1, 28, 28))
        probability = chainer.cuda.to_cpu(F.softmax(prediction).data[0])
        np.set_printoptions(precision=20, floatmode='fixed', suppress=True)
        ans = self.dnn.get_output()
        ans["output"] = probability
        #print(probability)
        #print(self.dnn.get_output())
        return ans

    def predict_rnd(self):
        #print(np.array(self.test[0]).shape)
        input = cp.random.randint(0,2,(1, 1, 28, 28)).astype(cp.float32)
        #print(input)
        #print(type(input))
        prediction = self.model.predictor(input)
        probability = chainer.cuda.to_cpu(F.softmax(prediction).data[0])
        np.set_printoptions(precision=20, floatmode='fixed', suppress=True)
        ans = self.dnn.get_output()
        ans["output"] = probability
        #print(probability)
        #print(self.dnn.get_output())
        return ans

    def predict_P_rnd(self):
        #print(np.array(self.test[0]).shape)

        n = 28
        w = 5
        b = np.random.randint(0, 10000)
        amp = 60

        input = []
        #print(b,x,y)
        for i in range(n):
            for j in range(n):
                x = np.random.random()
                y = np.random.random()
                pn = noise.pnoise2(i * x, j * y, octaves=w, base = b)#np.random.random()
                pn = self.f(pn, i, j)
                #print(pn)
                input.append(pn)#
        input = np.array(input).reshape((28,28))
        input = 1/(1+np.exp(-amp*input))
        input = self.normalize(input)

        input = cp.asarray(input).reshape((1,1,28,28)).astype(cp.float32)
        #print(input)
        #print(type(input))
        prediction = self.model.predictor(input)
        probability = chainer.cuda.to_cpu(F.softmax(prediction).data[0])
        np.set_printoptions(precision=20, floatmode='fixed', suppress=True)
        ans = self.dnn.get_output()
        ans["output"] = probability
        #print(probability)
        #print(self.dnn.get_output())
        return ans

    def normalize(self, i):
        max = np.max(i.flatten())
        min = np.min(i.flatten())
        return (i-min)/(max-min)

    def f(self, x, i, j):
        y = x - math.sqrt((i-14.5)**2+(j-14.5)**2)/50
        return y

    def getTrain_size(self):
        return len(self.train)
    def getTest_size(self):
        return len(self.test)
    def getTrain(self, n):
        return self.train[n][0][0]

    def get_dnn(self):
        return self.dnn
    def get_model(self):
        return self.model

    def getOutput_size(self,layer = ""):
        prediction = self.predict_Test(0)
        sizes = {}
        for k in prediction.keys():
            sizes[k] = prediction[k].shape
        if layer != "":
            sizes = sizes[layer]
        return sizes

    def getTestAnswer(self, num):
        return self.test[num][1]

class SmoothGrad:
    def __init__(self, model, dnn, sigma, d, output_part, size, path, input_type):
        self.train, self.test = datasets.get_mnist(ndim=3)
        self.model = model
        self.dnn = dnn
        self.d = d
        self.sigma = sigma
        self.output_part = output_part
        self.size = size
        self.path = path
        self.it_i = input_type

    def create_Input(self, n):
        input = []
        if self.it_i == "normal":
            input = self.train[n][0][0]
        elif self.it_i == "rnd":
            input = cp.random.randint(0,2,(1, 1, 28, 28))
        elif self.it_i == "Prnd":
            n = 28
            w = 5
            b = np.random.randint(0, 10000)
            amp = 60

            #print(b,x,y)
            for i in range(n):
                for j in range(n):
                    x = np.random.random()
                    y = np.random.random()
                    pn = noise.pnoise2(i * x, j * y, octaves=w, base = b)#np.random.random()
                    pn = self.f(pn, i, j)
                    #print(pn)
                    input.append(pn)#
            input = np.array(input).reshape((28,28))
            input = 1/(1+np.exp(-amp*input))
            input = self.normalize(input)

            input = cp.asarray(input).reshape((1,1,28,28)).astype(cp.float32)
        input = self.addNoise(input)
        return input

    def predict_Train_withNoise(self, i, num):
        #print(np.array(self.test[0]).shape)
        #print(n)
        #print(i.dtype)
        #self.createImage(n)
        #print(cp.min(input),cp.max(input))
        prediction = self.model.get_model().predictor(i.astype(cp.float32))
        probability = F.softmax(prediction).data[0]
        ans = self.dnn.get_output()
        ans["output"] = probability
        ans = ans[self.output_part].reshape(self.size, -1)
        return [num, ans]

        """
        ansD = self.predict_Train_withNoise_D(i)
        grad = cp.ndarray((self.size,28,28))
        for i in range(ans.shape[0]):
            grad[i] = ((ansD[i]-ans[i])/self.d).reshape((28,28))
        #print(probability)
        #print(self.dnn.get_output())
        return grad
        """

    def predict_Train_withNoise_D(self, i, x, y, num):
        inputD = cp.copy(i)
        inputD[0,0,x,y] += self.d
        predictionD = self.model.get_model().predictor(inputD.astype(cp.float32))
        probabilityD = F.softmax(predictionD).data[0]
        ans_pD = self.dnn.get_output()
        ans_pD["output"] = probabilityD
        """
        if x==0 and y==0:
            ans_pD2 = ans_pD[self.output_part].reshape(self.size, -1)
        else:
            ans_pD2 = cp.hstack((ans_pD2, ans_pD[self.output_part].reshape((self.size, -1))))
        """
        return [num,x,y,cp.asnumpy(ans_pD[self.output_part].reshape(self.size, -1))]

    def addNoise(self,data):
        return cp.array(data)+cp.random.normal(0, self.sigma, (28, 28)).astype(cp.float32).reshape(1, 1, 28, 28)

    def createImage(self, num):
        i = self.input.flatten()
        max = cp.max(i)
        min = cp.min(i)
        a = np.uint8(cp.asnumpy((self.input-min)/(max-min)) * 255).reshape(28,28)
        img = Image.fromarray(a)
        img.save(self.path+"\\input-"+str(num)+".jpg")

