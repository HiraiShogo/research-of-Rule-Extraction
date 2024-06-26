import torch.nn as nn
import torch.nn.functional as F
import torch

import sys


class RNN1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, device):
        # 初期隠れ状態の初期化
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # RNNの出力
        out, _ = self.rnn(x, h0)
        # 最後のタイムステップの出力を使用して分類
        out = self.fc(out[:, -1, :])
        return out


class RNN2(nn.Module):
    def __init__(self):
        super(RNN2, self).__init__()

        self.seq_len = 28  # 画像の Height を時系列のSequenceとしてRNNに入力する
        self.feature_size = 28  # 画像の Width を特徴量の次元としてRNNに入力する
        self.hidden_layer_size = 50  # 隠れ層のサイズ
        self.rnn_layers = 1

        self.rnn1 = nn.RNN(input_size=self.feature_size,
                                 hidden_size=self.hidden_layer_size,
                                 num_layers=self.rnn_layers)
        self.fc1 = nn.Linear(self.hidden_layer_size,10)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10,10)

    def init_hidden(self, batch_size):  # RNNの隠れ層 hidden を初期化
        #hedden = torch.randn(self.rnn_layers, batch_size, self.hidden_layer_size)
        hedden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hedden

    def forward(self, x):
        batch_size = x.shape[0]

        #self.hidden = self.init_hidden(batch_size)
        #hedden = torch.randn(self.rnn_layers, batch_size, self.hidden_layer_size)

        x = x.view(batch_size, self.seq_len,
                   self.feature_size)  # (Batch, Cannel, Height, Width) -> (Batch, Height, Width) = (Batch, Seqence, Feature)
        # 画像の Height を時系列のSequenceに、Width を特徴量の次元としてRNNに入力する
        x = x.permute(1, 0, 2)  # (Batch, Seqence, Feature) -> (Seqence , Batch, Feature)

        rnn_out, h_n = self.rnn1(x)  # RNNの入力データのShapeは(Seqence, Batch, Feature)
        # (h_n) のShapeは (num_layers, batch, hidden_size)
        x = h_n[-1, :, :]

        x = x.view(x.size()[0], -1)

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


class Cnn1(nn.Module):
    #x_=[]
    def __init__(self):
        super(Cnn1,self).__init__()
        #self.relu1=nn.ReLU()
        #self.soft=nn.Softmax()
        self.pool=nn.MaxPool2d(2,stride=2)
        #self.pool2=nn.MaxPool2d(2,stride=2)

        self.conv1=nn.Conv2d(1,10,3)
        self.conv2=nn.Conv2d(10,10,3)
        self.conv3=nn.Conv2d(10,10,3)
        self.conv4=nn.Conv2d(10,10,3)
        self.conv5=nn.Conv2d(10,10,3)
        self.conv6=nn.Conv2d(10,10,3)
        self.conv7=nn.Conv2d(10,10,3)
        #self.conv3=nn.Conv2d(10,10,16)

        self.fc1=nn.Linear(10,10)
        self.relu1=nn.ReLU()
        self.fc2=nn.Linear(10,10)
        #self.fc2=nn.Linear(120,10)


    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2,stride=2)
        x=self.conv3(x)
        x=F.relu(x)
        x=self.conv4(x)
        x=F.relu(x)
        x=self.conv5(x)
        x=F.relu(x)
        x=self.conv6(x)
        x=F.relu(x)
        x=self.conv7(x)
        x=F.relu(x)
        x=self.pool(x)
        x=x.view(x.size()[0],-1)
        x=self.fc1(x)
        x=self.relu1(x)
        x=self.fc2(x)

        return x

class Cnn2(nn.Module):
    def __init__(self):
        super(Cnn2,self).__init__()
        self.relu1=nn.ReLU()
        #self.soft=nn.Softmax()
        self.pool=nn.MaxPool2d(2,stride=2)

        self.conv1=nn.Conv2d(1,10,3)
        self.conv2=nn.Conv2d(10,10,3)
        self.conv3=nn.Conv2d(10,10,3)
        self.conv4=nn.Conv2d(10,20,3)
        self.conv5=nn.Conv2d(20,20,3)
        self.conv6=nn.Conv2d(20,20,3)
        self.conv7=nn.Conv2d(20,20,3)
        #self.conv3=nn.Conv2d(10,10,16)

        self.fc1=nn.Linear(20,20)
        self.fc2=nn.Linear(20,10)

        #self.fc2=nn.Linear(120,10)


    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2,stride=2)
        x=self.conv3(x)
        x=F.relu(x)
        x=self.conv4(x)
        x=F.relu(x)
        x=self.conv5(x)
        x=F.relu(x)
        x=self.conv6(x)
        x=F.relu(x)
        x=self.conv7(x)
        x=F.relu(x)
        x=self.pool(x)
        x=x.view(x.size()[0],-1)
        x=self.fc1(x)
        x=self.relu1(x)
        x=self.fc2(x)

        return x



class Cnn3(nn.Module):
    def __init__(self):
        super(Cnn3,self).__init__()
        self.relu1=nn.ReLU()
        #self.soft=nn.Softmax()
        self.pool=nn.MaxPool2d(2,stride=2)

        self.conv1=nn.Conv2d(1,10,3)
        self.conv2=nn.Conv2d(10,10,3)
        self.conv3=nn.Conv2d(10,10,3)
        self.conv4=nn.Conv2d(10,20,3)
        self.conv5=nn.Conv2d(20,20,3)
        self.conv6=nn.Conv2d(20,30,3)
        self.conv7=nn.Conv2d(30,30,3)
        #self.conv3=nn.Conv2d(10,10,16)

        self.fc1=nn.Linear(30,20)
        self.fc2=nn.Linear(20,10)

        #self.fc2=nn.Linear(120,10)


    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2,stride=2)
        x=self.conv3(x)
        x=F.relu(x)
        x=self.conv4(x)
        x=F.relu(x)
        x=self.conv5(x)
        x=F.relu(x)
        x=self.conv6(x)
        x=F.relu(x)
        x=self.conv7(x)
        x=F.relu(x)
        x=self.pool(x)
        x=x.view(x.size()[0],-1)
        x=self.fc1(x)
        x=self.relu1(x)
        x=self.fc2(x)

        return x

class Cnn4(nn.Module):
    def __init__(self):
        super(Cnn4,self).__init__()
        self.relu1=nn.ReLU()
        #self.soft=nn.Softmax()
        self.pool=nn.MaxPool2d(2,stride=2)

        self.conv1=nn.Conv2d(1,10,3)
        self.conv2=nn.Conv2d(10,10,3)
        self.conv3=nn.Conv2d(10,10,3)
        self.conv4=nn.Conv2d(10,10,3)
        self.conv5=nn.Conv2d(10,10,3)
        self.conv6=nn.Conv2d(10,10,3)
        self.conv7=nn.Conv2d(10,15,3)
        #self.conv3=nn.Conv2d(10,10,16)

        self.fc1=nn.Linear(15,20)
        self.fc2=nn.Linear(20,10)
        #self.fc2=nn.Linear(120,10)


    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2,stride=2)
        x=self.conv3(x)
        x=F.relu(x)
        x=self.conv4(x)
        x=F.relu(x)
        x=self.conv5(x)
        x=F.relu(x)
        x=self.conv6(x)
        x=F.relu(x)
        x=self.conv7(x)
        x=self.pool(x)
        x=x.view(x.size()[0],-1)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)

        return x

class Cnn5(nn.Module):
    #x_=[]
    def __init__(self):
        super(Cnn5,self).__init__()
        #self.relu1=nn.ReLU()
        #self.soft=nn.Softmax()
        self.pool=nn.MaxPool2d(2,stride=2)
        #self.pool2=nn.MaxPool2d(2,stride=2)

        self.conv1=nn.Conv2d(1,10,3)
        self.conv2=nn.Conv2d(10,10,3)
        self.conv3=nn.Conv2d(10,10,3)
        self.conv4=nn.Conv2d(10,10,3)
        self.conv5=nn.Conv2d(10,10,3)
        self.conv6=nn.Conv2d(10,10,3)
        self.conv7=nn.Conv2d(10,11,3)
        #self.conv3=nn.Conv2d(10,10,16)

        self.fc1=nn.Linear(11,20)
        self.fc2=nn.Linear(20,11)
        #self.fc2=nn.Linear(120,10)


    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2,stride=2)
        x=self.conv3(x)
        x=F.relu(x)
        x=self.conv4(x)
        x=F.relu(x)
        x=self.conv5(x)
        x=F.relu(x)
        x=self.conv6(x)
        x=F.relu(x)
        x=self.conv7(x)
        x=self.pool(x)
        x=x.view(x.size()[0],-1)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)

        return x
    
class Cnn9(nn.Module):
    #x_=[]
    def __init__(self):
        super(Cnn9,self).__init__()
        #self.relu1=nn.ReLU()
        #self.soft=nn.Softmax()
        self.pool=nn.MaxPool2d(2,stride=2)
        #self.pool2=nn.MaxPool2d(2,stride=2)

        self.conv1=nn.Conv2d(1,10,3)
        self.conv2=nn.Conv2d(10,10,3)
        self.conv3=nn.Conv2d(10,10,3)
        self.conv4=nn.Conv2d(10,10,3)
        self.conv5=nn.Conv2d(10,10,3)
        self.conv6=nn.Conv2d(10,10,3)
        self.conv7=nn.Conv2d(10,20,3)
        #self.conv3=nn.Conv2d(10,10,16)

        self.fc1=nn.Linear(20,20)
        self.fc2=nn.Linear(20,10)
        #self.fc2=nn.Linear(120,10)


    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2,stride=2)
        x=self.conv3(x)
        x=F.relu(x)
        x=self.conv4(x)
        x=F.relu(x)
        x=self.conv5(x)
        x=F.relu(x)
        x=self.conv6(x)
        x=F.relu(x)
        x=self.conv7(x)
        x=F.sigmoid(x)
        x=self.pool(x)
        x=x.view(x.size()[0],-1)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)

        return x
    
class Cnn10(nn.Module):
    #x_=[]
    def __init__(self):
        super(Cnn10,self).__init__()
        #self.relu1=nn.ReLU()
        #self.soft=nn.Softmax()
        self.pool=nn.MaxPool2d(2,stride=2)
        #self.pool2=nn.MaxPool2d(2,stride=2)

        self.conv1=nn.Conv2d(1,10,3)
        self.conv2=nn.Conv2d(10,10,3)
        self.conv3=nn.Conv2d(10,10,3)
        self.conv4=nn.Conv2d(10,10,3)
        self.conv5=nn.Conv2d(10,10,3)
        self.conv6=nn.Conv2d(10,10,3)
        self.conv7=nn.Conv2d(10,16,3)
        #self.conv3=nn.Conv2d(10,10,16)

        self.fc1=nn.Linear(16,20)
        self.fc2=nn.Linear(20,10)
        #self.fc2=nn.Linear(120,10)


    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2,stride=2)
        x=self.conv3(x)
        x=F.relu(x)
        x=self.conv4(x)
        x=F.relu(x)
        x=self.conv5(x)
        x=F.relu(x)
        x=self.conv6(x)
        x=F.relu(x)
        x=self.conv7(x)
        x=F.sigmoid(x)
        x=self.pool(x)
        x=x.view(x.size()[0],-1)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)

        return x
    
class Cnn11(nn.Module):
    #x_=[]
    def __init__(self):
        super(Cnn11,self).__init__()
        #self.relu1=nn.ReLU()
        #self.soft=nn.Softmax()
        self.pool=nn.MaxPool2d(2,stride=2)
        #self.pool2=nn.MaxPool2d(2,stride=2)

        self.conv1=nn.Conv2d(1,10,3)
        self.conv2=nn.Conv2d(10,10,3)
        self.conv3=nn.Conv2d(10,10,3)
        self.conv4=nn.Conv2d(10,10,3)
        self.conv5=nn.Conv2d(10,10,3)
        self.conv6=nn.Conv2d(10,10,3)
        self.conv7=nn.Conv2d(10,30,3)
        #self.conv3=nn.Conv2d(10,10,16)

        self.fc1=nn.Linear(30,20)
        self.fc2=nn.Linear(20,10)
        #self.fc2=nn.Linear(120,10)


    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2,stride=2)
        x=self.conv3(x)
        x=F.relu(x)
        x=self.conv4(x)
        x=F.relu(x)
        x=self.conv5(x)
        x=F.relu(x)
        x=self.conv6(x)
        x=F.relu(x)
        x=self.conv7(x)
        x=F.sigmoid(x)
        x=self.pool(x)
        x=x.view(x.size()[0],-1)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)

        return x