'''
Created on 2019/07/01

@author: ail
'''
"""
from tkinter import *
from tkinter import ttk
import tkinter
"""
from dnn_learn import Model, DNN, SmoothGrad  # @UnresolvedImport
from logic_extract import Logic  # @UnresolvedImport
from image_extract import Image_save  # @UnresolvedImport
from PIL import Image
from chainer import serializers
import chainer.links as L
import datetime
import os
import threading
from multiprocessing import Pool, Value
import wx
import glob
import numpy as np
import re
import subprocess
import cupy as cp
import time

#==============================
#変数設定
#
#並列計算
pallarel_p = True #前処理
pallarel_l = True #論理式抽出
pallarel_i = True #SmoothGrad
pallarel_m = False #統合
#
#Logic Extract:
enable_l = True
Units_l = 60000
input_type_l = "normal" #rnd:乱数入力, normal:学習データ入力
input_part = "Pool2"
output_part = "output"
#
#Image Extract(SmoothGrad):
enable_i = True
Units_i = 10000
input_type_i = "normal" #rnd:乱数入力, normal:学習データ入力
sigma = 0.1
d = 0.004
#
#Merge
enable_m = False
#==============================

def arg_wrapper(args):
    return args[0](*args[1:])

#論理式抽出：データの取得→dnn_learn.py
def L_data(model, input_type_l, n):
    #print(n)
    if input_type_l == "normal":
        return model.predict_Train(n)
    elif input_type_l == "rnd":
        return model.predict_rnd()
    elif input_type_l == "Prnd":
        return model.predict_P_rnd()

#SmoothGrad：入力データの取得→dnn_learn.py
def I_data(sg, n):
    #print(n)
    return sg.create_Input(n)

#論理式の抽出→logic_extract.py
def L_extract(L, n):
    return L.extract(n)

#抽出した論理式の精度計算→logic_extract.py
def L_test(L, n):
    return L.test(n)

#SmoothGrad：出力データの計算その1→dnn_learn.py
def I_culc(sg, n, num):
    #print(n)
    return sg.predict_Train_withNoise(n, num)

#SmoothGrad：出力データの計算その2→dnn_learn.py
def I_culcD(sg, n, x, y, num):
    #print(n)
    return sg.predict_Train_withNoise_D(n, x, y, num)

#SmoothGrad：出力データの整形その1
def I_makei(input, size):
    i = cp.ndarray((Units_i, size, 1))
    for j in input:
        i[j[0]] = cp.asarray(j[1])
    return i

#SmoothGrad：出力データの整形その2
def I_makeid(i, size):
    id = np.ndarray((Units_i, size,28,28))
    for j in i:
        arr = j[3]
        for k in range(len(arr)):
            id[j[0], k, j[1], j[2]] = arr[k,0]
    return id

#SmoothGrad：勾配計算
def I_culcG(i, id):
    grad = cp.ndarray((i.shape[0], 28,28))
    loop = id.shape
    for k in range(loop[0]):
        for l in range(loop[1]):
            for m in range(loop[2]):
                grad[k,l,m] = ((cp.asarray(id[k,l,m])-i[k,0])/d)
    return grad

#論理式とSmoothGrad出力の統合
def Merge(u, f, image, image_R):
    ans = cp.ones(image.shape)
    for s in f.split("∧"):
        if s[0] == "P":
            ans = cp.minimum(ans, image)
        else:
            ans = cp.minimum(ans, image_R)
    return [u, f, ans]

#正規化
def normalize(i):
        max = cp.max(i.flatten())
        min = cp.min(i.flatten())
        return (i-min)/(max-min)

#画像生成：論理式ごと
def createImage_f(u, rank, image, path):
    a1 = normalize(image)
    a2 = np.uint8(cp.asnumpy(a1) * 255)
    #print(a2.shape)
    img = Image.fromarray(a2)
    img.save(path+"\\normal\\rank"+str(rank+1)+'.jpg')
    img.resize((168,168)).save(path+"\\zoom\\rank"+str(rank+1)+'_zoom.jpg')

#統合：ORの計算
def Merge_inUnit(u,image):
    ans = cp.zeros(image[0][1].shape)
    for i in image:
        ans = cp.maximum(ans, i[1])
    return [u,ans]

#重み付き平均による統合
def Merge_inUnit_ave(u,image, logic):
    ans = cp.zeros(image[0][1].shape)
    sum = 0
    for i in image:
        sum += logic[i[0]]
        ans += i[1]*logic[i[0]]
    return [u, ans/sum]

#画像生成：出力素子ごと
def createImage_u(image, path):
    a1 = normalize(image[1])
    a2 = np.uint8(cp.asnumpy(a1) * 255)
    #print(a2.shape)
    img = Image.fromarray(a2)
    img.save(path+"\\normal\\unit-"+str(image[0])+'.jpg')
    img.resize((168,168)).save(path+"\\zoom\\unit-"+str(image[0])+'_zoom.jpg')

#時間文字列の整形
def time_laber(start, end):
    label = ""
    time_l = end - start
    if time_l >= 3600*24:
        day = int(time_l/86400)
        label += str(day)+" day"
        if day != 1:
            label += "s "
        else:
            label += " "
        time_l %= 86400
    if time_l >= 3600:
        hour = int(time_l/3600)
        label += str(hour)+" hour"
        if hour != 1:
            label += "s "
        else:
            label += " "
        time_l %= 3600
    elif len(label) != 0:
        label += "0 hour "
    if time_l >= 60:
        min = int(time_l/60)
        label += str(min)+" min"
        if min != 1:
            label += "s"
        else:
            label += " "
        time_l %= 60
    elif len(label) != 0:
        label += "0 min "
    label += str(round(time_l, 2)) +" sec(s)"
    return label

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

    #コンソール出力を.txtに
    console = open(path + "\\console_output.txt", "w")

    #論理式の各パス・フォルダの生成・実験条件の出力
    path_l = path+"\\logic"
    path_f = path_l+"\\logic_formula"
    path_r = path_l+"\\process_result"
    if enable_l:
        os.makedirs(path_f)
        os.makedirs(path_r)
        w = open(path_l+"\\Attribute.txt", "w")
        w.write("Units = " + str(Units_l)+"\n")
        w.write("  → input type = " + input_type_l+"\n")
        w.write("input_part = "+input_part+"\n")
        w.write("output_part = "+output_part)
        w.close()

    #SmoothGradの各パス・フォルダの生成・実験条件の出力
    path_i = path+"\\image"
    path_t = path_i+"\\train"
    path_n = path_i+"\\normal"
    path_noise = path_i+"\\noise"
    path_z = path_i+"\\zoom"
    path_s = path_i+"\\simplize"
    path_sz = path_i+"\\simplize_zoom"
    if enable_i:
        os.makedirs(path_n)
        #os.makedirs(path_t)
        os.makedirs(path_z)
        os.makedirs(path_s)
        os.makedirs(path_noise)
        os.makedirs(path_sz)
        w = open(path_i+"\\Attribute.txt", "w")
        w.write("Units = "+str(Units_i)+"\n")
        w.write("  → input type = " + input_type_l+"\n")
        w.write("sigma = "+str(sigma)+"\n")
        w.write("d = "+str(d)+"\n")
        w.write("output_part = "+input_part)
        w.close()

    #統合の各パス・フォルダの生成
    path_m = path + "\\merge"
    if enable_m:
        os.makedirs(path_m)

    #モデルの初期化
    model = None
    pathN = None
    app = wx.App()
    #論理式/SmoothGradのどちらかをやるならネットワークファイル(.npz)を読み込む
    if enable_l or enable_i:
        filter = "network file(*npz) | *.npz| All file(*.*) | *.*"
        dialog = wx.FileDialog(None, u'ネットワークファイルを選択してください', '', '', filter)
        dialog.ShowModal()
        pathN = dialog.GetPath()
        model = Model()
        model.load_Network(pathN)
        console.write("network file : "+pathN+"\n\n")

    #統合を行う、かつ論理式/SmoothGradのどちらかを行わないなら、使うデータの選択
    path_all = ""
    if enable_m:
        if not(enable_l or enable_i):
            dialog = wx.DirDialog(None, u'使う結果フォルダを選択してください',path)
            dialog.ShowModal()
            path_all = dialog.GetPath()
            console.write("result folder : "+path_all+"\n\n")

    print("Initialize Complete!\n")

    #使う変数の設定
    path_l = ""
    path_i = ""
    path_a = ""
    argument_L = []
    argument_I = []
    argument_ID = []
    argument_LT = []
    L = Logic(model, path, input_part, output_part, input_type_l)
    I = None


    start = time.time()

    #前処理：論理式→各入力に対する出力を計算する
    if enable_l:
        print("Preprocessing_Part : Initializing for Logic_Part ...", end = "", flush=True)
        console.write("Preprocessing_Part : Initializing for Logic_Part ...")
        argument_outL = [(L_data, model, input_type_l, each) for each in range(Units_l)]
        out_L = []
        #model.predict_P_rnd()
        if pallarel_p:
            pool = Pool()
            out_L = pool.map(arg_wrapper, argument_outL)
            pool.close()
        else:
            for a in argument_outL:
                out_L.append(arg_wrapper(a))
        argument_L = [(L_extract, L, each) for each in out_L]
        argument_LT = [(L_test, L, each) for each in range(model.getTest_size())]
        print("  → finished", flush=True)
        console.write("  → finished\n")

    #前処理：SmoothGrad→使う入力を作る
    if enable_i:
        print("Preprocessing_Part : Initializing for Image_Part ...", end = "", flush=True)
        console.write("Preprocessing_Part : Initializing for Image_Part ...")
        I = Image_save(model, path, input_part)
        size = int(cp.prod(cp.asarray(model.getOutput_size(input_part))))
        sg = SmoothGrad(model, model.get_dnn(), sigma, d, input_part, size, path, input_type_i)
        argument_outI = [(I_data, sg, each) for each in range(Units_i)]
        #model.predict_P_rnd()
        input_I = []
        if pallarel_p:
            pool = Pool()
            input_I = pool.map(arg_wrapper, argument_outI)
            pool.close()
        else:
            for a in argument_outI:
                input_I.append(arg_wrapper(a))
        argument_I = [(I_culc, sg, input_I[each], each) for each in range(len(input_I))]
        argument_ID = [(I_culcD, sg, input_I[each], x, y, each) for each in range(len(input_I)) for x in range(28) for y in range(28)]
        print("  → finished", flush=True)
        console.write("  → finished\n")

    end = time.time()
    #print(out_L)
    if enable_i or enable_l:
        label = time_laber(start, end)
        print("Preprocessing_Part : Complete !", flush=True)
        console.write("Preprocessing_Part : Complete !\n")
        print("     execution time = "+label+"\n", flush=True)
        console.write("     execution time = "+label+"\n\n")

    grads = []

    #論理式パート
    if enable_l:
        start_l = time.time()

        #出力からルールの抽出
        print("Logic_Part : Extracting rules ...", end="", flush=True)
        console.write("Logic_Part : Extracting rules ...")
        output = []
        if pallarel_l:
            pool = Pool()
            output = pool.map(arg_wrapper, argument_L)
            pool.close()
        else:
            for a in argument_L:
                output.append(arg_wrapper(a))
        print("  → finished", flush=True)
        console.write("  → finished\n")

        #ルールをデータベースに追加
        print("Logic_Part : Adding to database ...", end="", flush=True)
        console.write("Logic_Part : Adding to database ...")
        L.add_Logics(output)
        print("  → finished", flush=True)
        console.write("  → finished\n")

        #出現回数の多い順にソート(やんなくても動作するかも)
        print("Logic_Part : Sorting rules ...", end="", flush=True)
        console.write("Logic_Part : Sorting rules ...")
        L.sort_output(path_f)
        print("  → finished", flush=True)
        console.write("  → finished\n")

        #ルールの精度計算
        print("Logic_Part : Testing rules ...", end="", flush=True)
        console.write("Logic_Part : Testing rules ...")
        ts = []
        if pallarel_l:
            pool = Pool()
            ts = pool.map(arg_wrapper, argument_LT)
            pool.close()
        else:
            for a in argument_LT:
                ts.append(arg_wrapper(a))
        print("  → finished", flush=True)
        console.write("  → finished\n")

        #精度計算の結果を出力
        L.end(model.getTest_size(), ts)
        end_l = time.time()
        print("Logic_Part : Complete !", flush=True)
        console.write("Logic_Part : Complete !\n")
        label = time_laber(start_l, end_l)
        print("     execution time = "+label+"\n", flush=True)
        console.write("     execution time = "+label+"\n\n")

    #論理式を抽出しない、かつ統合はする
    elif enable_m:
        #結果フォルダから論理式を読み込む
        print("Logic_Part : Loading rules ...", end="", flush=True)
        console.write("Logic_Part : Loading rules ...")
        L.load_Logics(path_all)
        print("  → finished", flush=True)
        console.write("  → finished\n\n")

    #画像パート
    if enable_i:
        start_i = time.time()

        #入力から出力を計算
        print("Image_Part : Culculating output ...", end="", flush=True)
        console.write("Image_Part : Culculating output ...")
        output = []
        if pallarel_i:
            pool = Pool()
            output = pool.map(arg_wrapper, argument_I)
            pool.close()
        else:
            for a in argument_I:
                output.append(arg_wrapper(a))
        print("  → finished", flush=True)
        console.write("  → finished\n")

        #データを使える形に整形
        print("Image_Part : Transforming output data ...", end="", flush=True)
        console.write("Image_Part : Transforming output data ...")
        out_I = I_makei(output, size)
        print("  → finished", flush=True)
        console.write("  → finished\n")
        output = []

        #微小値を加えた入力から出力を計算
        print("Image_Part : Culculating output with delta ...", end="", flush=True)
        console.write("Image_Part : Culculating output with delta ...")
        out_ID = []
        if pallarel_i:
            pool = Pool(int(os.cpu_count()/2))
            out_ID = pool.map(arg_wrapper, argument_ID)
            pool.close()
        else:
            for a in argument_ID:
                out_ID.append(arg_wrapper(a))
        print("  → finished", flush=True)
        console.write("  → finished\n")

        #データを使える形に整形
        print("Image_Part : Transforming output data with delta ...", end="", flush=True)
        console.write("Image_Part : Transforming output data with delta ...")
        out_ID = I_makeid(out_ID, size)
        print("  → finished", flush=True)
        console.write("  → finished\n")

        #数値偏微分(勾配)を計算
        print("Image_Part : Culculating gradients ...", end="", flush=True)
        console.write("Image_Part : Culculating gradients ...")
        argument_IG = [(I_culcG, out_I[each], out_ID[each]) for each in range(len(out_I))]
        grad = []
        if pallarel_i:
            pool = Pool(int(os.cpu_count()/2))
            grad = pool.map(arg_wrapper, argument_IG)
            pool.close()
        else:
            for a in argument_IG:
                grad.append(arg_wrapper(a))
        out_I = []
        out_ID = []
        grads = cp.zeros(grad[0].shape)
        for i in grad:
            grads += i
        print("  → finished", flush=True)
        console.write("  → finished\n")

        #画像化→image_extract.py
        print("Image_Part : Creating gradients' images ...", end="", flush=True)
        console.write("Image_Part : Creating gradients' images ...")
        I.save_ImageI(Units_i, input_I, path_t, path_noise)
        I.save_ImageG(grads, path_n, path_s, path_z, path_sz)
        print("  → finished", flush=True)
        console.write("  → finished\n")

        end_i = time.time()
        label = time_laber(start_i, end_i)
        print("Image_Part : Complete !", flush=True)
        console.write("Image_Part : Complete !\n")
        print("     execution time = "+label+"\n", flush=True)
        console.write("     execution time = "+label+"\n\n")

    #SmoothGradはやらないが、統合はやる
    elif enable_m:
        #結果フォルダからSmoothGradの結果を読み込む
        print("Image_Part : Loading images ...", end="", flush=True)
        console.write("Image_Part : Loading images ...")
        image = []
        for i in glob.glob(path_all+'\\image\\normal\\*'):
            im = Image.open(i)
            image.append(np.array(im))
        grads = cp.asarray(np.array(image))/255
        #print(grads)
        print("  → finished", flush=True)
        console.write("  → finished\n\n")

    #統合パート
    if enable_m:
        logics = L.get_Logics()
        pathes = []
        for p in range(len(logics)):
            pathes.append(path_m+"\\unit-"+str(p))
            os.makedirs(path_m+"\\unit-"+str(p)+"\\normal")
            os.makedirs(path_m+"\\unit-"+str(p)+"\\zoom")
        grads_R = 1-grads
        argument_M = []
        images = []
        for e1 in range(len(logics)):
            for e2 in logics[e1].keys():
                argument_M.append((Merge, e1, e2, grads[e1], grads_R[e1]))
        if pallarel_m:
            pool = Pool()
            images = pool.map(arg_wrapper, argument_M)
            pool.close()
        else:
            for a in argument_M:
                images.append(arg_wrapper(a))

        argument_IF = [(createImage_f, e[0], list(logics[e[0]].keys()).index(e[1]), e[2], pathes[e[0]]) for e in images]
        if pallarel_m:
            pool = Pool()
            pool.map(arg_wrapper, argument_IF)
            pool.close()
        else:
            for a in argument_IF:
                arg_wrapper(a)

        merged = []
        for i in range(len(logics)):
            merged.append([])
        for i in images:
            merged[i[0]].append([i[1],i[2]])

        path_mn = path_m+"\\all\\normal"
        path_ma = path_m+"\\all\\average"
        os.makedirs(path_mn+"\\normal")
        os.makedirs(path_mn+"\\zoom")
        os.makedirs(path_ma+"\\normal")
        os.makedirs(path_ma+"\\zoom")
        argument_IU = [(Merge_inUnit, e, merged[e]) for e in range(len(logics))]
        argument_IUA = [(Merge_inUnit_ave, e, merged[e],logics[e]) for e in range(len(logics))]
        image_inUnit = []
        image_inUnit_ave = []
        if pallarel_m:
            pool = Pool()
            image_inUnit = pool.map(arg_wrapper, argument_IU)
            image_inUnit_ave = pool.map(arg_wrapper, argument_IUA)
            pool.close()
        else:
            for a in argument_IU:
                image_inUnit.append(arg_wrapper(a))
            for a in argument_IUA:
                image_inUnit_ave.append(arg_wrapper(a))

        argument_IU = [(createImage_u, e, path_mn) for e in image_inUnit]
        argument_IUA = [(createImage_u, e, path_ma) for e in image_inUnit_ave]
        if pallarel_m:
            pool = Pool()
            pool.map(arg_wrapper, argument_IU)
            pool.map(arg_wrapper, argument_IUA)
            pool.close()
        else:
            for a in argument_IU:
                arg_wrapper(a)
            for a in argument_IUA:
                arg_wrapper(a)

    console.close()
    cmd = path
    subprocess.Popen(['explorer',cmd])
    print("All Finished !")

