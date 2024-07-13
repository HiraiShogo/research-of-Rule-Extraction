# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *  # @UnresolvedImport
from common.gradient import numerical_gradient  # @UnresolvedImport
import wx


class ConvNet:
    """単純なConvNet

    conv - relu - pool - affine - relu - affine - softmax

    Parameters
    ----------
    input_size : 入力サイズ（MNISTの場合は784）
    hidden_size_list : 隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
    output_size : 出力サイズ（MNISTの場合は10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
        'relu'または'he'を指定した場合は「Heの初期値」を設定
        'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
    """
    def __init__(self, input_dim=(1, 28, 28)):
        """
                 conv_param={'filter_num':10, 'filter_size':5, 'pad':0, 'stride':1},
                 conv2_param={'filter2_num':10, 'filter2_size':5, 'pad2':0, 'stride2':1},
                 conv3_param={'filter3_num':10, 'filter3_size':5, 'pad3':0, 'stride3':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        self.filter_num = conv_param['filter_num']
        self.filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        self.filter2_num = conv2_param['filter2_num']
        self.filter2_size = conv2_param['filter2_size']
        filter2_pad = conv2_param['pad2']
        filter2_stride = conv2_param['stride2']
        self.filter3_num = conv3_param['filter3_num']
        self.filter3_size = conv3_param['filter3_size']
        filter3_pad = conv3_param['pad3']
        filter3_stride = conv3_param['stride3']
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        """
        self.struct = ["start"]
        self.lookup = {}
        self.WBs = []
        self.sizes = {"start":input_dim}
        self.counter = {"conv":0,"pool":0,"affine":0,"relu":0,"sigmoid":0}
        self.L_param = {}
        self.wb_count = 1
        """
        conv_output_size = (seinput_size - self.filter_size + 2*filter_pad) / filter_stride + 1
        conv_output_size2 = (int(conv_output_size - self.filter2_size + 2*filter2_pad) / filter2_stride) + 1
        conv_output_size3 = (int(conv_output_size2 - self.filter3_size + 2*filter3_pad) / filter3_stride) + 1
        pool_output_size = int(self.filter3_num * (conv_output_size3/2) * (conv_output_size3/2))
        print(pool_output_size)
        """

        # 重みの初期化
        self.params = {}
        """
        self.params['W1'] = weight_init_std * \
                            np.random.randn(self.filter_num, input_dim[0], self.filter_size, self.filter_size)
        self.params['b1'] = np.zeros(self.filter_num)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(self.filter2_num, self.filter_num, self.filter2_size, self.filter2_size)
        self.params['b2'] = np.zeros(self.filter2_num)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(self.filter3_num, self.filter2_num, self.filter3_size, self.filter3_size)
        self.params['b3'] = np.zeros(self.filter3_num)
        self.params['W4'] = weight_init_std * \
                            np.random.randn(pool_output_size, self.hidden_size)
        self.params['b4'] = np.zeros(self.hidden_size)
        self.params['W5'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b5'] = np.zeros(output_size)
        """

        # レイヤの生成
        self.layers = OrderedDict()
        """
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'],
                                           conv2_param['stride2'], conv2_param['pad2'])
        self.layers['Relu2'] = Relu()
        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'],
                                           conv3_param['stride3'], conv3_param['pad3'])
        self.layers['Relu3'] = Sigmoid()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['Sigmoid1'] = Sigmoid()
        self.layers['Affine2'] = Affine(self.params['W5'], self.params['b5'])

        self.last_layer = SoftmaxWithLoss()
        """

    def addLayer(self, layer = "", param = []):
        weight_init_std=0.01
        layer = layer.lower()

        if layer == "conv":#param = (素子数, サイズ, pad, stride)
            self.params['W'+str(self.wb_count)] = weight_init_std * \
                            np.random.randn(param[0], self.sizes[self.struct[len(self.struct)-1]][0], param[1], param[1])
            self.params['b'+str(self.wb_count)] = np.zeros(param[0])
            name = 'Conv'+str(self.counter["conv"]+1)
            self.layers[name] = Convolution(self.params['W'+str(self.wb_count)], self.params['b'+str(self.wb_count)],
                                           param[3], param[2], 'W'+str(self.wb_count), 'b'+str(self.wb_count))
            self.counter["conv"] += 1
            self.lookup[name] = ("W"+str(self.wb_count), "b"+str(self.wb_count))
            self.wb_count += 1
            self.WBs.append(name)
            input_shape = (self.sizes[self.struct[len(self.sizes)-1]][1], self.sizes[self.struct[len(self.sizes)-1]][2])
            h,w = self.layers[name].getOutput_num(input_shape)
            self.sizes[name] = (param[0],h,w)


        elif layer == "pool":#param = (h, w, stride)
            name = 'Pool'+str(self.counter["pool"]+1)
            self.layers[name] = Pooling(pool_h=param[0], pool_w=param[1], stride=param[2])
            self.counter["pool"] += 1
            input_shape = (self.sizes[self.struct[len(self.sizes)-1]][1], self.sizes[self.struct[len(self.sizes)-1]][2])
            h,w = self.layers[name].getOutput_num(input_shape)
            self.sizes[name] = (self.sizes[self.struct[len(self.sizes)-1]][0],h,w)

        elif layer == "affine":#param = (素子数)
            name = 'Affine'+str(self.counter["affine"]+1)
            size = 0
            if "Pool" in self.struct[len(self.sizes)-1]:
                size = self.sizes[self.struct[len(self.sizes)-1]][0] * self.sizes[self.struct[len(self.sizes)-1]][1]*self.sizes[self.struct[len(self.sizes)-1]][2]
            else:
                size = self.sizes[self.struct[len(self.sizes)-1]]
            self.params['W'+str(self.wb_count)] = weight_init_std * \
                            np.random.randn(size, param[0])
            self.params['b'+str(self.wb_count)] = np.zeros(param[0])
            self.layers[name] = Affine(self.params['W'+str(self.wb_count)], self.params['b'+str(self.wb_count)], 'W'+str(self.wb_count), 'b'+str(self.wb_count))
            self.counter["affine"] += 1
            self.lookup[name] = ("W"+str(self.wb_count), "b"+str(self.wb_count))
            self.wb_count += 1
            self.WBs.append(name)
            self.sizes[name] = param[0]

        elif layer == "relu":
            name = 'ReLU'+str(self.counter["relu"]+1)
            self.layers[name] = Relu()
            self.counter["relu"] += 1
            self.sizes[name] = self.sizes[self.struct[len(self.struct)-1]]

        elif layer == "sigmoid":
            name = 'Sigmoid'+str(self.counter["sigmoid"]+1)
            self.layers[name] = Sigmoid()
            self.counter["sigmoid"] += 1
            self.sizes[name] = self.sizes[self.struct[len(self.struct)-1]]

        elif layer == "end":
            name = "SoftmaxWithLoss"
            self.last_layer = SoftmaxWithLoss()
            #print(self.params.keys())
            for i in self.struct:
                print(i+" :")
                if "Conv" in i or "Affine" in i:
                    print("  "+self.lookup[i][0]+", "+self.lookup[i][1])
                print("  "+str(self.sizes[i])+"\n")

        if  np.sum(self.sizes[name][1:]) == 0 or np.sum(self.sizes[name]) != np.sum(np.abs(self.sizes[name])):
            print("invalid value \"MINUS or ZERO\" : in " + name +" ,"+str(self.sizes[name]))
            sys.exit()
        self.struct.append(name)
        self.L_param[name] = param
        #print(self.params.keys())
        #print(name + " : " +str(self.sizes[self.struct[len(self.sizes)-1]]))

    def loadNetwork(self, to = "" , path = ""):
        if path == "":
            app = wx.App()
            filter = "network file(*.pkl) | *.pkl| All file(*.*) | *.*"
            dialog = wx.FileDialog(None, u'ネットワークのpickleファイルを選択してください', '', '', filter)
            dialog.ShowModal()
            path = dialog.GetPath()
        #print("ImageFolder: "+pathI)

        with open(path, 'rb') as f:
            network = pickle.load(f)

        #print(network.keys())

        for i in network['struct']:
            name = i
            #print(i)
            att = network[i]
            Wb = 0
            if ("Conv" in i or "Affine" in i):
                Wb = network[i+"Wb"]
                self.params[Wb[0]] = network[Wb[0]]
                self.params[Wb[1]] = network[Wb[1]]
            if "Conv" in i:
                self.layers[name] = Convolution(self.params[Wb[0]], self.params[Wb[1]],
                                           att[3], att[2], Wb[0], Wb[1])
            elif "Pool" in i:
                self.layers[name] = Pooling(pool_h=att[0], pool_w=att[1], stride=att[2])

            elif "Affine" in i:
                self.layers[name] = Affine(self.params[Wb[0]], self.params[Wb[1]], Wb[0], Wb[1])

            elif "ReLU" in i:
                self.layers[name] = Relu()

            elif "Sigmoid" in i:
                self.layers[name] = Sigmoid()
            elif i == "SoftmaxWithLoss":
                self.layers["Softmax"] = Softmax()

            #print(self.layers)
            if to == i:
                break

    def showNetwork(self):
        print(self.layers.keys())

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """損失関数を求める
        引数のxは入力データ、tは教師ラベル
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        """勾配を求める（数値微分）

        Parameters
        ----------
        x : 入力データ
        t : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in self.WBs:
            grads[self.lookup[idx][0]] = numerical_gradient(loss_w, self.params[self.lookup[idx][0]])
            grads[self.lookup[idx][1]] = numerical_gradient(loss_w, self.params[self.lookup[idx][1]])

        return grads

    def gradient(self, x, t):
        """勾配を求める（誤差逆伝搬法）

        Parameters
        ----------
        x : 入力データ
        t : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for i in self.WBs:
            #print(self.lookup[i])
            grads[self.lookup[i][0]], grads[self.lookup[i][1]] = self.layers[i].dW, self.layers[i].db
        """
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Conv3'].dW, self.layers['Conv3'].db
        grads['W4'], grads['b4'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W5'], grads['b5'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        """
        return grads

    def save_params(self, file_name="Myparams.pkl", acc = 0, todaydetail = ""):
        #os.makedirs("params\\"+str(todaydetail))
        params = {}
        for key, val in self.params.items():
            params[key] = val
        params["struct"] = self.struct[1:]
        for key, val in self.lookup.items():
            params[key+"Wb"] = val
        for key, val in self.L_param.items():
            params[key] = val
        for key, val in self.sizes.items():
            params[key+"_S"] = val
        with open("params\\"+str(todaydetail)+"\\"+file_name, 'wb') as f:
            pickle.dump(params, f)
        fw = open(("params\\"+str(todaydetail)+"\\setting.txt"), "w")
        fw.write("Structure:\n")

        fw.write("train_acc = " + str(acc[1])+"\n")
        fw.write("test_acc = " + str(acc[0]))

    def load_params(self, file_name="Myparams.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Conv2', 'Conv3', 'Affine1', 'Affine2']):#'Affine2'
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]