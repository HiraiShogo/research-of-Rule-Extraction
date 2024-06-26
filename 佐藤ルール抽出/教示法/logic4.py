'''
Created on 2018/06/01

@author: ail
'''

import glob
from convolutional_network import ConvNet# @UnresolvedImport
#from cnn_57.convolutional_networkM2 import SimpleConvNet2# @UnresolvedImport
from dataset.mnist import load_mnist  # @UnresolvedImport
from common.functions import softmax  # @UnresolvedImport
from booleanCulc import booleanCulc as bc  # @UnresolvedImport
from booleanCulcM import booleanCulcM as bcM  # @UnresolvedImport
import datetime
import time
import os
import re
import pickle
import sys
import subprocess
import numpy as np
import random as rnd

Units = -1
Flag = True #True:Notあり False:Notなし
del_num = -1 #(この数)番目までの頻度の論理式を使う
mode = "N" #N：通常モード M：多値モード

if __name__ == '__main__':
    print("Initializing...")
    with open("cnn_412\\params.pkl", 'rb') as f:
            network = pickle.load(f)
    print(network.keys())

    #プーリング出力
    network = ConvNet(input_dim=(1,28,28))
    """
    conv_param = {'filter_num': 20, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        conv2_param = {'filter2_num': 20, 'filter2_size': 8, 'pad2': 0, 'stride2': 1},
                        conv3_param = {'filter3_num': 20, 'filter3_size': 16, 'pad3': 0, 'stride3': 1},
                        hidden_size=20, output_size=10, weight_init_std=0.01
    """
    #出力層出力(Affineまで)
    network_out = ConvNet(input_dim=(1,28,28))
    """
    ,
                        conv_param = {'filter_num': 20, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        conv2_param = {'filter2_num': 20, 'filter2_size': 8, 'pad2': 0, 'stride2': 1},
                        conv3_param = {'filter3_num': 20, 'filter3_size': 16, 'pad3': 0, 'stride3': 1},
                        hidden_size=20, output_size=10, weight_init_std=0.01
    """
    network.loadNetwork("Pool1")
    network_out.loadNetwork()

    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    out_num = network.predict(x_train[0].reshape((1,1,28,28))).shape[1]
    dataList = []
    testList = []
    if Units == -1:
        dataList = x_train
        testList = x_test
    else:
        dataList = x_train[:Units]
        if Units > len(x_test):
            testList = x_test
    print("Initialize Finished\n")

    print("Input folder Name")
    todaydetail = input(' name > ')
    if len(todaydetail) == 0:
        todaydetail = str(datetime.datetime.today()).replace(".", "_").replace(":","_")
    os.makedirs("logic\\"+str(todaydetail))
    os.makedirs("logic\\"+str(todaydetail)+"\\test_result")

    fw = open(("logic\\"+str(todaydetail)+"\\attributes.txt"), "w")
    fw.write("Units = "+str(Units)+"\n")
    fw.write("del_num = "+str(del_num)+"\n")

    logic = []
    for i in range(out_num):
        logic.append({})

    print("Culculating...")
    for pic in range(len(dataList)):
        pool = network.predict(dataList[pic].reshape((1,1,28,28)))
        pool = pool.flatten()
        pool = pool*2
        pool -= pool*(pool-1)/2
        pool = pool.astype(np.int)
        #print(pool)

        out = network_out.predict(dataList[pic].reshape((1,1,28,28))).flatten()
        #out = softmax(out.flatten())
        out = out * 2
        out -= out*(out-1)/2
        out = out.astype(np.int)
        #print(str(out)+"\n\n")
        #print(out)

        #print(pool)
        #print(str(out) + "  " + str(t_train[pic]) + "\n")

        for num in range(len(out)):
            #print(out)
            if(out[num] == 1):
                l = ""
                s = 0
                #print(pool)
                for num2 in range(len(pool)):
                    if pool[num2] == 1:
                        if s == 0:
                            l += "P_x"+str(num2)
                            s += 1
                        else:
                            l += " AND P_x"+str(num2)

                for num2 in range(len(pool)):
                    if pool[num2] == 0:
                        if s == 0:
                            l += "N_x"+str(num2)
                            s += 1
                        else:
                            l += " AND N_x"+str(num2)

                #if num == 0:
                #  print(l)
                if l not in logic[num].keys():
                    logic[num][l] = 1
                else:
                    logic[num][l] += 1
        if pic*100%len(dataList) == 0:
            print(str(pic*100/len(dataList))+"% Finished")

    print("Finished Culculation\n")

    print("Sorting Now...")
    sorted = []
    s_num = -1
    for list in logic:
        sorted.append([])
        s_num += 1
        for key, value in list.items():
            #print("key: "+key+"  value: "+str(value))
            if len(sorted[s_num]) == 0:
                sorted[s_num].append(key)
                #print("appendFirst")
            else:
                for i in range(len(sorted[s_num])):
                    if value > list[sorted[s_num][i]]:
                        sorted[s_num].insert(i,key)
                        #print("insert:"+str(i))
                        break
                    if i==len(sorted[s_num])-1 and value <= list[sorted[s_num][i]]:
                        sorted[s_num].append(key)
                        #print("appendLast")

    use = []
    for s in sorted:
        if len(s)>del_num:
            use.append(s[0:del_num])
            #print(s[0:del_num])
        else:
            use.append(s)
            #print(s)
    print("Sorting Finished\n")

    print("Creating \"test_result.txt\"...")
    fwm = open(("logic\\"+str(todaydetail)+"\\まとめ.txt"), "w")
    bool = []
    fwm.write("*************************\n論理式\n*************************\n")
    for i in range(len(use)):
        fw = open(("logic\\"+str(todaydetail)+"\\test_result"+"\\result["+ str(i) + "].txt"), "w")
        fwm.write("  "+str(i)+"：\n")
        for j in use[i]:
            fw.write(str(j)+", 頻度:"+str(logic[i][j])+"\n")
            fwm.write(str("    ")+str(j)+", 頻度:"+str(logic[i][j])+"\n")
        fw.write("\n***********************\n")
        for j in range(len(use[i]),len(sorted[i])):
            fw.write(str(sorted[i][j])+". 頻度："+str(logic[i][sorted[i][j]])+"\n")
        if mode == "N":
            bool.append(bc(use[i]))
        elif mode == "M":
            bool.append(bcM(use[i]))
        fwm.write("\n")
    print("Finished creating\n")

    fwm.write("*************************\n論理式(Ofiice数式対応)\n*************************\n")
    for i in range(len(use)):
        fwm.write("  "+str(i) +"：\n")
        for j in use[i]:
            o = j.replace(' AND ', '∧')
            o = re.sub(r'P_x(\d*)', r'x_\1', o)
            o = re.sub(r'N_x(\d*)', r'(x_\1)\\bar', o)
            fwm.write("    "+str(o)+", 頻度："+str(logic[i][j])+"\n")
        fwm.write("\n")


    print("Testing extracted_Rules...")
    sum = 0
    fw = open(("logic\\"+str(todaydetail)+"\\acc_process.txt"), "w")
    for i in range(len(testList)):
        pool = network.predict(testList[i].reshape((1,1,28,28)))
        pool = pool.flatten()
        pool = pool*2
        pool -= pool*(pool-1)/2
        pool = pool.astype(np.int)

        r = []
        ans = np.zeros((out_num))
        ans[t_test[i]] = 1

        for b in bool:
            r.append(b.culc(pool))
        r = np.array(r)

        if mode == "M":
            r = softmax(r)
            r = r*2
            r -= r*(r-1)/2
            r = r.astype(np.int)

        isOk = int(np.dot(ans,np.array(r)))
        allsum = np.sum(np.array(r))
        if allsum != 1:
            isOk = 0
        sum += isOk

        fw.write("論理式出力: "+str(r)+"  test: "+str(t_test[i])+"   正解or不正解: "+str(isOk)+"\n")

    fw.write("\n=====================================\n")
    fw.write("test: "+str(len(testList))+" データ,  正解数: "+str(sum)+"\n")
    fw.write("     →正解率: "+str(sum*100/len(testList))+"％\n")

    fwm.write("*************************\nテスト結果\n*************************\n")
    fwm.write("test: "+str(len(testList))+" データ,  正解数: "+str(sum)+"\n")
    fwm.write("     →正解率: "+str(sum*100/len(testList))+"％\n")

    print("All Finished")

    fw.close()


    cmd = 'logic\\'+str(todaydetail)
    subprocess.Popen(['explorer',cmd])
    fw.close()

