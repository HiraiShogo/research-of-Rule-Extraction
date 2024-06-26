'''
Created on 2018/06/01

@author: ail

データ構造
  (任意フォルダ)
        ー(抽出したルールのフォルダ)
            ーmiddles
               ー0～
            ーout
               ー0～
        ーlogic3.py
        ーconvolutional_networkM1_B
        ーconvolutional_networkM2_B
                   ・
                   ・
                   ・

'''

import glob
from convolutional_networkM1_B import SimpleConvNet# @UnresolvedImport
from convolutional_networkM2_B import SimpleConvNet2# @UnresolvedImport
from bc_layer import bc_layer  # @UnresolvedImport
from dataset.mnist import load_mnist  # @UnresolvedImport
from common.functions import sigmoid  # @UnresolvedImport
import datetime
import time
import os
import re
import sys
import subprocess
from bool_culc import bool_culc as bc  # @UnresolvedImport
import numpy as np
import random as rnd

path1 = 'CNN_M\\out'
path2 = 'CNN_M\\middles'
Units = 1000
Flag = True #True:Notあり False:Notなし

if __name__ == '__main__':
    #出力層入力
    network1 = SimpleConvNet(input_dim=(1,28,28),
                        conv_param = {'filter_num': 10, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        conv2_param = {'filter2_num': 10, 'filter2_size': 8, 'pad2': 0, 'stride2': 1},
                        conv3_param = {'filter3_num': 10, 'filter3_size': 16, 'pad3': 0, 'stride3': 1},
                        hidden_size=10, output_size=10, weight_init_std=0.01)
    #中間層1入力
    network2 = SimpleConvNet2(input_dim=(1,28,28),
                        conv_param = {'filter_num': 10, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        conv2_param = {'filter2_num': 10, 'filter2_size': 8, 'pad2': 0, 'stride2': 1},
                        conv3_param = {'filter3_num': 10, 'filter3_size': 16, 'pad3': 0, 'stride3': 1},
                        hidden_size=10, output_size=10, weight_init_std=0.01)

    networks = [network1, network2]

    #MNIST読み込み
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    dataList = x_test[:Units]
    """
    a = network2.predict(x_train[0].reshape((1,1,28,28)))
    x1 = rnd.randint(0,len(a[0,0])-1)
    y1 = rnd.randint(0,len(a[0,0,x1])-1)
    """

    #結果フォルダ名前つけ
    files = glob.glob(path1+'\\*')
    todaydetail = input(' name > ')
    if len(todaydetail) == 0:
        todaydetail = str(datetime.datetime.today()).replace(".", "_").replace(":","_")
    os.makedirs("logic\\"+str(todaydetail))
    best_logic = []
    for i in files:
        best_logic.append("")
    start = time.time()

    for file in files:
        with open(file+'\\logic_output.txt') as f: #論理式読み込み
            logic = f.read()

        number = int(str(file).replace(path1,'').replace('\\',''))#[-2]
        os.makedirs("logic\\"+str(todaydetail)+ "\\out\\" + str(number))
        fw = open(('logic\\'+str(todaydetail) + "\\out\\" + str(number) + "\\result.txt"), "w")
        fw.write("\'Not\' enabled: "+str(Flag)+"\n\n")
        output = 0

        acc = 0
        max = 0
        for i in logic.split('\n'): #論理式1つずつループ
            if(len(i)!=0):
                #print(i)
                fw.write(i+':\n')
                sum = 0
                for u in range(0, Units): #MNISTデータループ
                    result = 1
                    #print("  "+str(u)+"回目")
                    #中間層出力
                    network_output = networks[0].predict(dataList[u].reshape((1,1,28,28)))
                    network_output = network_output.reshape((len(network_output[0])))
                    #論理式計算
                    b = bc(param = i, flag = Flag)
                    result = b.culc((network_output*2).astype(np.int))
                    #if result == 1:
                        #print("number = " + str(number) + "   ans = "+str(t_test[u]))

                    #print("    result = "+str(result) +"  t_test = "+str(t_test[u])+ "  number : "+str(number)+"\n")
                    if(result==1 and number == t_test[u]):
                        sum += 1
                    elif(result==0 and number != t_test[u]):
                        sum += 1
                    #print(u)
                #print()
                acc = sum*100/Units
                if acc>max:
                    best_logic[number] = i
                    max = acc
                #print(sum)
                fw.write('acc : '+str(acc)+"\n")
                fw.write("===========================================\n\n")
        print("file"+str(number)+" finished")

    os.makedirs("logic\\"+str(todaydetail)+ "\\out\\all")
    fw = open(('logic\\'+str(todaydetail) + "\\out\\all\\result.txt"), "w")
    fw.write("\'Not\' enabled: "+str(Flag)+"\n\n")
    for u in range(0, Units): #出力層総合確認
        for i in best_logic:
            result = 1
            l = 0
            network_output = networks[0].predict(dataList[u].reshape((1,1,28,28)))#[0,int(arg2[1])-1,x1,y1]
            #print("  "+str(u)+"回目")
            for j in i.split(' AND '):
                arg2 = j.replace('_x','')
                if(network_output[0,int(arg2[1:])]>=0.5):#,x1,y1
                    output = 1
                else :
                    output = 0
            #print("    "+str(arg2)+", " + str(output)+" -> "+(arg2[0]=='P' and str(output) or str(abs(output-1))))
                if(arg2[0]=='P'):
                    result *= output
                else:
                    if(Flag):
                        result *= abs(output-1)
                #fw.write("    result = "+str(result) +"  t_test = "+str(t_test[u])+ "  number : "+str(number)+"\n")
            if(result==1 and number == t_test[u]):
                l += 1
            elif(result==0 and number != t_test[u]):
                l += 1
                    #print(u)
                #print()
        if l == len(best_logic):
            sum += 1
    acc = sum*100/Units
    fw.write('acc : '+str(acc)+"\n")
    fw.write("===========================================\n\n")

    #ファイルに出力
    for i in range(0,len(best_logic)):
        fw.write(str(i)+": "+best_logic[i]+"\n")

    #出力層のみ
    if not os.path.exists(path2):
        sys.exit()
    files = glob.glob(path2+'\\*')
    num = 0

    #出力層レイヤ生成
    bcol = bc_layer()
    for i in best_logic:
        bcol.addItem(i, Flag)

    nw = []
    mid_layer = []

    #中間層

    for f1 in files: #層数ループ
        mid = glob.glob(f1+'\\*')
        number = int(str(f1).replace(path2+'\\',''))
        if len(mid) == 0:
            break
        nw.append([])
        mid_layer.append(bc_layer())
        for f2 in mid: #素子ループ
            num2 = int(str(f2).replace(f1,'').replace('\\',''))
            #print(num2)
            os.makedirs("logic\\"+str(todaydetail)+ "\\middle\\" + str(number) + "\\" + str(num2))#[-2]
            fw = open(('logic\\'+str(todaydetail) + "\\middle\\" + str(number) + "\\" + str(num2) + "\\result.txt"), "w")#[-2]
            fw.write("\'Not\' enabled: "+str(Flag)+"\n\n")
            with open(f2+"\\logic_output.txt") as f:
                l = f.read()
            l2 = l.split('\n')

            acc = 0
            max = 0
            b = "null"

            if len(l) != 0:
                for i in l2:
                    if len(i)==0:
                        break

                    fw.write(i+" : \n")
                    test_f = bc(param = i, flag = Flag)
                    sum = 0
                    for u in range(0, Units): #MNISTデータループ
                        result = 1
                        #print("  "+str(u)+"回目")
                        network_output = networks[number-1].predict(dataList[u].reshape((1,1,28,28)))
                        network_output2 = networks[number].predict(dataList[u].reshape((1,1,28,28)))
                        #print(network_output2.shape)
                        network_output2 = network_output2.reshape(network_output2.shape[0], -1)
                        #print(network_output2)
                        #データの整理
                        no2 = []
                        #print("network_output = "+str(network_output2))
                        for a in network_output2.reshape((len(network_output2[0]))):#[0]
                            if(a>=0.5):#[x1,y1]
                                no2.append(1)
                            else:
                                no2.append(0)

                        #論理式出力
                        out = test_f.culc(no2)

                        #ニューラルネットワーク出力
                        network_output = network_output.reshape((len(network_output[0])))
                        network_output = (network_output*2).astype(np.int)
                        #print("network_output = "+str(network_output))
                        #print("result = "+str(network_output[int(f2[-2])])+"\n")

                        if result == 1:
                            print("no2 = "+str(no2))
                            print("out = "+str(out) + "\n")

                        if network_output[int(num2)] == out:
                            sum += 1
                        #print(u)
                    #print()
                    acc = sum*100/Units
                    print("acc = "+str(acc)+"\n\n")
                    if acc>max:
                        b = i
                        max = acc
                    fw.write('acc : '+str(acc)+"\n")
                    fw.write("===========================================\n\n")
            mid_layer[number-1].addItem(b,Flag) #終了→追加

    end = time.time()
    print("All finished - "+str(end - start)+"[sec]")
    os.makedirs("logic\\"+str(todaydetail)+ "\\middle\\" + str(number) + "\\all")
    fw = open(('logic\\'+str(todaydetail) + "\\middle\\" + str(number) + "\\all\\result.txt"), "w")
    print("********middles******")
    count = 0
    for i in mid_layer[0].getItem():
        print(i)
        fw.write(str(count)+" : "+i+"\n")
        count += 1
    print("\n********out******")
    for i in best_logic:
        print(i)

    #中間層総合確認
    fw = open(('logic\\'+str(todaydetail) + "\\middle\\result.txt"), "w")
    sum = 0
    for pic in range(0, Units): #MNISTデータループ
        input = []
        inp = networks[-1].predict(dataList[pic].reshape((1,1,28,28)))
        for a in inp.reshape(len(inp[0])):
            if(a>=0.5):
                input.append(1)
            else:
                input.append(0)
        print(input)
        for l in range(1,len(mid_layer)+1):
            input = mid_layer[len(mid_layer)-l].output(input)

        output  = bcol.output(input)
        print("out = "+str(output)+"    ans = "+str(t_test[pic])+"\n")
        fw.write("out = "+str(output)+"    ans = "+str(t_test[pic])+"\n\n")

        ok = True
        for o in range(0,len(output)):
            if (output[o] == 1 and t_test[pic] != o) or (output[o] == 0 and t_test[pic] == o):
                ok = False
        if ok:
            sum += 1
    fw.write("*************************\n\n")
    fw.write("acc = "+str(sum*100.0/Units)+"\n\n")
    fw.write("*************************\n\n")

    #終了後フォルダを開く
    cmd = 'logic\\'+str(todaydetail)
    subprocess.Popen(['explorer',cmd])
    fw.close()

