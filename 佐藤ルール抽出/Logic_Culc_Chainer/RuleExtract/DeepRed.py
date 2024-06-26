'''
Created on 2019/10/28

@author: ail
'''

from dnn_learn import Model, DNN  # @UnresolvedImport
from C4_5 import C4_5  # @UnresolvedImport
from chainer import serializers
import chainer.links as L
import datetime
import os
from multiprocessing import Pool, Value
import wx
import glob
import numpy as np
import re
import subprocess
import cupy as cp
import time

#################################
#変数設定
Units = 100
layers = ["Pool2", "Sigmoid2", "output"]
#################################

def removeDuplicated(T):
    removed = []
    for t in T:
        for u in t:
            if not (u in t):
                removed.appned(u)
    return T

def binarize(x):
    y = -2*(x**2)+3*x
    return y.astype(int)

def rule_Culc(x, T):
    (c, th) = T
    if x[c] >th:
        return 1
    else:
        return 0

if __name__ == '__main__':
    #結果フォルダの名前入力
    print("Input folder Name")
    todaydetail = input(' name > ')
    path = "result\\"+todaydetail
    while(os.path.exists(path)):
        print()
        print("This path has already used!")
        print("Input again.")
        todaydetail = input(' name > ')
        path = "result\\"+todaydetail
    print()
    if len(todaydetail) == 0:
        todaydetail = str(datetime.datetime.today()).replace(".", "_").replace(":","_")
    path = "result\\"+todaydetail
    os.makedirs(path)

    #モデルの初期化
    model = None
    pathN = None
    app = wx.App()
    #論理式/SmoothGradのどちらかをやるならネットワークファイル(.npz)を読み込む
    filter = "network file(*npz) | *.npz| All file(*.*) | *.*"
    dialog = wx.FileDialog(None, u'ネットワークファイルを選択してください', '', '', filter)
    dialog.ShowModal()
    pathN = dialog.GetPath()
    model = Model()
    model.load_Network(pathN)

    print("Preprocessing_Part : Culculating samples ... [", end = "", flush=True)
    predicts = []
    for u in range(Units):
        predicts.append(model.predict_Train(u))
        if (u+1) *10 % Units == 0:
            print("*", end = "", flush=True)
    print("  → finished", flush=True)

    print("Initialize Complete!\n")

    c4_5 = C4_5()
    rule = []
    for c in range(1):
        rule.append({})
        rh = []
        ro = (c, 0.5)
        T = None
        count = 0
        for j in reversed(range(len(layers[:-1]))):
            rh.append([])
            if layers[j+1] == "output":
                T= [ro]
            else:
                T = rh[j]
            T = removeDuplicated(T)

            for t in T:
                xy = []
                for predict in predicts:
                    x = predict[layers[j]]
                    x = (-2*(x**2)+3*x).astype(int)
                    y = rule_Culc(predict[layers[j+1]], t)
                    xy.append((x,y))
                rh[count].append(c4_5.generate(xy))
