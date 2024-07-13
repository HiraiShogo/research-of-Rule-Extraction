'''
Created on 2018/04/24

@author: ail
'''

from PIL import Image
import numpy as np
#import cupy as np
import threading
import os
import time
import datetime
import logging
from convolutional_network import ConvNet# @UnresolvedImport
from dataset.mnist import load_mnist  # @UnresolvedImport
from numpy.core.tests.test_mem_overlap import xrange
import subprocess
import wx


Units = 10000  #画像数
delta = 0.02 #偏微分差分
sigma = 0.1 #正規分布偏差
filter_num = 10
outputInitial = False

#image = np.zeros((filter_num,28,28))

imageList = [] #[入力画像番号][チャンネル = 0][横][縦]

class Imageculc((threading.Thread)):
    def __init__(self,imageList,num,path):
        super(Imageculc, self).__init__()
        self.network = ConvNet(input_dim=(1,28,28))
        self.network.loadNetwork("Pool1",path)
        """
        ,
                        conv_param = {'filter_num': 20, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        conv2_param = {'filter2_num': 20, 'filter2_size': 8, 'pad2': 0, 'stride2': 1},
                        conv3_param = {'filter3_num': 20, 'filter3_size': 16, 'pad3': 0, 'stride3': 1},
                        hidden_size=10, output_size=10, weight_init_std=0.01)
        """
        #self.size = self.network.predict(imageList[0].reshape((1,1,28,28)))[0,0].flatten().shape#
        #print(self.size)
        self.image = np.zeros((28,28))
        #self.x1 = 0 #rnd.randint(0,12)
        #self.y1 = 0 #rnd.randint(0,12)
        self.outputs = 0
        self.imageList = np.copy(imageList)
        self.imageListD = []
        self.max = 0.0
        self.min = 100
        self.r = []
        self.num = num

    def run(self):

        for j in xrange(0, Units):
            #print(network.predict(normalize(imageList)).shape)
            self.outputs = self.network.predict(imageList[j].reshape((1,1,28,28))).flatten()[self.num]
            #print(self.outputs)
            for k in xrange(0,28):
                for l in xrange(0,28):
                    self.imageListD = np.copy(imageList[j])
                    self.imageListD[0,k,l] += delta
                    self.outputsD = self.network.predict(self.imageListD.reshape((1,1,28,28))).flatten()[self.num]#
            #print(str(outputs) + "     "+str(outputsD))
            #df[num,j,k,l]
                    self.image[k,l] += (self.outputsD - self.outputs) / delta
            #if (self.outputs == self.outputsD).any():
               # print(("FilterNum: "  +str(self.num)+  "   PicNum:"+str(j)+"  out: " +str(self.outputs))+"  outD: "+str(self.outputsD))
            if j*100%len(imageList) == 0:
                print("Filter "+str(self.num)+" : "+str(j*100/len(imageList))+"% Finished")
    #for m in range(0,len(df[num])):
    #    image = image + df[num,m]
        self.image = self.image / Units #len(df[num])
        #print(self.image)
        self.normalize_N()
        self.image = np.uint8(self.image * 255)
        Image.fromarray(self.image).save("image\\" +todaydetail+"\\result\\result"+str(self.num)+".jpg")
        #createImageFile2(self.normalize(self.image), "result", self.num)
        print("Filter "+str(self.num)+" complete")
        return

    def normalize(self):
        self.max = 0.0
        self.r = self.image
        for i in range(0,len(self.r)):
            for k in range(0, len(self.r[i])):
                if type(self.r[i,k]) != 'int':
                    for l in range(0, len(self.r[i,k])):
                        if self.max<abs(self.r[i,k,l]):
                            self.max = abs(self.r[i,k,l])
                else:
                    if self.max<abs(self.r[i,k]):
                        self.max = abs(self.r[i,k])

            self.r = (self.r + self.max)/(2*self.max)
        return self.r

    def normalize_N(self):
        self.max = 0.0
        self.min = 100
        self.r = self.image
        for i in range(0,len(self.r)):
            for k in range(0, len(self.r[i])):
                if type(self.r[i,k]) != 'int':
                    for l in range(0, len(self.r[i,k])):
                        if self.max<self.r[i,k,l]:
                            self.max = self.r[i,k,l]
                        elif self.min > self.r[i,k,l]:
                            self.min = self.r[i,k,l]
                else:
                    if self.max<self.r[i,k]:
                        self.max = self.r[i,k]
                    elif self.min>self.r[i,k]:
                        self.min = self.r[i,k]

            self.r = (self.r - self.min)/(self.max - self.min)
        return self.r


def normalizeInput(img):
    for i in range(0,len(img)):
        for k in range(0, len(img[i,0])):
            for l in range(0, len(img[i,0,k])):
                if max<img[i,0,k,l]:
                    max = img[i,0,k,l]
                elif min > img[i,0,k,l]:
                    min = img[i,0,k,l]

        img[i,0] = (img[i,0]-min)/(max-min)
        max = 0.0
        min = 100.0
    return img

"""
#@cuda.jit('void(int32[:,:,:,:], int32)')
def culc(self,num):
    self.image = np.zeros((28,28))
    self.network = SimpleConvNet(input_dim=(1,28,28),
                        conv_param = {'filter_num': 10, 'filter_size': 27, 'pad': 0, 'stride': 1},
                        weight_init_std=0.01)

    x1 = 0 #rnd.randint(0,12)
    y1 = 0 #rnd.randint(0,12)


    for j in xrange(0, Units):
        #print(network.predict(normalize(imageList)).shape)
        self.outputs = self.network.predict(imageList[j].reshape((1,1,28,28)))[0,num,x1,y1]#
        for k in xrange(0,28):
            for l in xrange(0,28):
                self.imageListD = np.copy(imageList[j])
                self.imageListD[0,k,l] += delta
                self.outputsD = self.network.predict(self.imageListD.reshape((1,1,28,28)))[0,num,x1,y1]#
            #print(str(outputs) + "     "+str(outputsD))
            #df[num,j,k,l]
                self.image[k,l] += (self.outputsD - self.outputs) / delta
        logging.debug(("FilterNum: "  +str(num)+  "   PicNum:"+str(j)+"  out: " +str(self.outputs)))
    #for m in range(0,len(df[num])):
    #    image = image + df[num,m]
    self.image = self.image / Units #len(df[num])
    #print(image[num])
    createImageFile2(normalize(normalize,self.image), "result", num)
    print("Filter "+str(num)+" complete")
    return
"""
def createImageFile(img, sig):
    for i in range (0, len(img)):
        a = np.uint8(img[i,0] * 255)
        Image.fromarray(a).save("image\\" +todaydetail+"\\"+sig+'\\'+sig+str(i)+'.jpg')
    return

def createImageFile2(img, sig, n):
    #for i in range (0, len(img)):
    a = np.uint8(img * 255)
    Image.fromarray(a).save("image\\" +todaydetail+"\\"+sig+'\\'+sig+str(n)+'.jpg')
    return

if __name__ == '__main__':

    todaydetail = input(' name > ')
    if len(todaydetail) == 0:
        todaydetail = str(datetime.datetime.today()).replace(".", "_").replace(":","_")
    os.makedirs("image\\"+str(todaydetail)+"\\result")
    os.makedirs("image\\"+str(todaydetail)+"\\initial")
    os.makedirs("image\\"+str(todaydetail)+"\\noisy_initial")

    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

    fw = open(("image\\"+str(todaydetail)+"\\attributes.txt"), "w")
    fw.write("Units = "+str(Units)+"\n")
    fw.write("delta = "+str(delta)+"\n")
    fw.write("sigma = "+str(sigma)+"\n")

    imageList = x_train[:Units]#初期化
    if outputInitial:
        print("初期画像出力開始...")
        createImageFile(imageList , "initial")
    imageList += np.random.normal(0, sigma, (Units, 1, 28, 28))
        #imageList = normalizeInput(imageList)
    if outputInitial:
        createImageFile(imageList , "noisy_initial")
        print(imageList)

    #createImageFile(imageList , "initial")
    print("初期画像出力終了")

    print("調整開始...")

    app = wx.App()
    filter = "network file(*.pkl) | *.pkl| All file(*.*) | *.*"
    dialog = wx.FileDialog(None, u'ネットワークのpickleファイルを選択してください', '', '', filter)
    dialog.ShowModal()
    path = dialog.GetPath()

    network = ConvNet(input_dim=(1,28,28))
    network.loadNetwork("Pool1",path)
    size = len(network.predict(imageList[0].reshape((1,1,28,28))).flatten())
    print(size)

    funcs = []
    for i in xrange(0, size):
        funcs.append(Imageculc(imageList,i,path))
        funcs[len(funcs)-1].start()
        #funcs[len(funcs)-1].join()
"""""
    flag = True
    while(flag):
        for f in funcs:
            flag = flag or f.is_alive()

    cmd = 'image\\'+str(todaydetail)
    subprocess.Popen(['explorer',cmd])
"""
"""
    for i in range(0,filter_num):
        print(str(i) + ":")
        for j in range(0, Units):
            #print(network.predict(normalize(imageList)).shape)
            outputs = network.predict(imageList)[j,i]#,x1,y1
            for k in range(0,28):
                for l in range(0,28):
                    imageListD = np.copy(imageList)
                    imageListD[j,0,k,l] = imageListD[j,0,k,l] + delta
                    outputsD = network.predict(imageListD)[j,i]#,x1,y1
                    #print(str(outputs) + "     "+str(outputsD))
                    df[j,k,l] = (outputsD - outputs) / delta
            print("    PicNum:"+str(j)+"  out: " +str(outputs))

        for m in range(0,len(df)):
            image[i,0] = image[i,0] + df[m]
        image[i,0] = image[i,0] / len(df)

    print("出力画像出力開始...")
        #imageList = normalize(imageList)
        #print(image)
    createImageFile(image, "result")
    print("終了")
"""

