import torch.nn as nn
import torch.nn.functional as F

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