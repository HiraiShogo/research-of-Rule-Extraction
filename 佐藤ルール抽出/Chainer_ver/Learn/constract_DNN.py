'''
Created on 2019/10/15

@author: ail
'''
import os
import sys
import shutil
import wx

###################################
mode = "lib" #talk:CUIの対話モード、lib:ライブラリモード

###########################
#Write netowork
#  c : 畳み込み層、("c", (出力数), (フィルタサイズ), (ストライド default=1), (パディング default=0))
#  p : プーリング層、("p", (フィルタサイズ default=2), (パディング default=0))
#  a : 全結合層、("a", (出力数))
#  s : シグモイド、("s")
#  r : ReLU、("r")
#end : ネットワーク終了、("end")
net = [
    ("c", 5, 9),
    ("p"),
    ("c", 10, 9),
    ("s"),
    ("p"),
    ("a", 10),
    ("s"),
    ("a", 10),
    ("s"),
    ("a", 10),
    ("end")
       ]
###########################

if __name__ == '__main__':
    name = input("input DNN_name : ")
    path = "networks\\"+name+".py"
    while(os.path.exists(path)):
        print()
        print("This name has already used!")
        print("Input again.")
        name = input("input DNN_name : ")
        path = "networks\\"+name+".py"
    print()

    """
    data_s = input("input dataset : ").lower()
    input_size = []
    if "mnist" in data_s:
        input_size = [28,28]
        print("input size = "+str(input_size)+"\n")
    """
    input_size = (28,28)

    input_num = 1
    network = []
    nums = {"Conv":0, "Pool":0, "Affine":0, "Sigmoid":0, "ReLU":0}
    struct = open("struct.txt",'w')
    args = {}
    error = False

    #対話モード
    if mode == "talk":
        while True:
            layer = input("  layer : ").lower()
            if "c" in layer:
                print("  Conv"+str(nums["Conv"]+1)+" :")
                struct.write(("Conv"+str(nums["Conv"]+1)+" :\n"))
                fs = int(input("    filter size : "))
                struct.write("    filter size : "+str(fs)+"\n")
                output_size = int(input("    >> output size : "))
                struct.write("    output size : "+str(output_size)+"\n")
                stride_s = input("    >> stride(default=1) : ")
                stride = 1
                if stride_s != "":
                    stride = int(stride_s)
                    struct.write("    stride : "+str(stride)+"\n")
                padding_s = input("    >> padding(default=0) : ")
                padding = 0
                if padding_s != "":
                    padding = int(padding_s)
                    struct.write("    padding : "+str(padding)+"\n")

                H,W = input_size[0],input_size[1]
                FH,FW = fs, fs
                out_h = 1 + int((H + 2*padding - FH) / stride)
                out_w = 1 + int((W + 2*padding - FW) / stride)
                if out_h * out_w <= 0:
                    print("ERROR : Conv"+str(nums["Conv"]+1)+" : output size is zero or minus !")
                    struct.write("ERROR : Conv"+str(nums["Conv"]+1)+" : output size is zero or minus !")
                    error = True
                    struct.close()
                    sys.exit(0)
                input_size = [out_w, out_h]

                network.append("Conv"+str(nums["Conv"]+1))
                nums["Conv"] += 1
                args["Conv"+str(nums["Conv"])] = (input_num, output_size, fs, stride, padding)
                input_num = output_size
                print("    *output_size = "+str(input_size))
                struct.write("    *output_size = "+str(input_size)+"\n")
                print("  Conv"+str(nums["Conv"])+"Added!")
                print()
                struct.write("\n")

            elif "p" in layer:
                print("  Pool"+str(nums["Pool"]+1)+" :")
                struct.write("Pool"+str(nums["Pool"]+1)+" :\n")
                fs_s = input("    >> filter size(default=2) : ")
                fs = 2
                if fs_s != "":
                    fs = int(fs_s)
                    struct.write("    filter size : "+str(fs)+"\n")
                padding_s = input("    >> padding(default=0) : ")
                padding = 0
                if padding_s != "":
                    padding = int(padding_s)
                    struct.write("    padding : "+str(padding)+"\n")
                H,W = input_size[0], input_size[1]
                out_h = int(1 + (H - fs) / fs)
                out_w = int(1 + (W - fs) / fs)
                if out_h * out_w <= 0:
                    print("ERROR : Pool"+str(nums["Pool"]+1)+" : output size is zero or minus !")
                    struct.write("ERROR : Pool"+str(nums["Pool"]+1)+" : output size is zero or minus !")
                    error = True
                    struct.close()
                    sys.exit(0)
                input_size = [out_w, out_h]
                network.append("Pool"+str(nums["Pool"]+1))
                nums["Pool"] += 1
                args["Pool"+str(nums["Pool"])] = (fs, padding)
                print("    *output_size = "+str(input_size))
                struct.write("    *output_size = "+str(input_size)+"\n")
                print("  Pool"+str(nums["Pool"])+"Added!")
                print()
                struct.write("\n")

            elif "a" in layer:
                print("  Affine"+str(nums["Affine"]+1)+" :")
                struct.write("Affine"+str(nums["Affine"]+1)+" :\n")
                network.append("Affine"+str(nums["Affine"]+1))
                nums["Affine"] += 1
                output_size = int(input("    >> output size : "))
                struct.write("    output size : "+str(output_size)+"\n")
                args["Affine"+str(nums["Affine"])] = (input_num, output_size)
                input_num = output_size
                print("  Affine"+str(nums["Affine"])+"Added!")
                print()
                struct.write("\n")

            elif "s" in layer:
                print("  Sigmoid"+str(nums["Sigmoid"]+1)+" :")
                struct.write("Sigmoid"+str(nums["Sigmoid"]+1)+" :\n")
                network.append("Sigmoid"+str(nums["Sigmoid"]+1))
                nums["Sigmoid"] += 1
                args["Sigmoid"+str(nums["Sigmoid"])] = None
                print("  Sigmoid"+str(nums["Sigmoid"])+"Added!")
                print()
                struct.write("\n")

            elif "r" in layer:
                print("  ReLU"+str(nums["ReLU"]+1)+" :")
                struct.write("ReLU"+str(nums["ReLU"]+1)+" :")
                network.append("ReLU"+str(nums["ReLU"]+1))
                nums["ReLU"] += 1
                args["ReLU"+str(nums["ReLU"])] = None
                print("  ReLU"+str(nums["ReLU"])+"Added!")
                print()

            elif "end" in layer:
                print("  Constracted !")
                struct.write("end")
                break
    #ライブラリモード
    elif mode == "lib":
        if net[-1] != "end":
            print("Uncorrect Network")
            sys.exit(0)

        for n in net:
            layer = n[0]
            if "c" in layer:
                if len(n)<3:
                    print("Conv Layer needs \"2\" Arguments at Least")
                    sys.exit(0)
                print("  Conv"+str(nums["Conv"]+1)+" :")
                fs = n[2]
                output_size = n[1]
                struct.write("Conv"+str(nums["Conv"]+1)+" :\n")
                struct.write("    filter_size : "+str(fs)+"\n")
                struct.write("    output_size : "+str(output_size)+"\n")
                stride = 1
                if len(n)>=4:
                    stride = n[3]
                    struct.write("    stride : "+str(stride)+"\n")
                padding = 0
                if len(n)>=5:
                    padding = n[4]
                    struct.write("    padding : "+str(padding)+"\n")

                H,W = input_size[0],input_size[1]
                FH,FW = fs, fs
                out_h = 1 + int((H + 2*padding - FH) / stride)
                out_w = 1 + int((W + 2*padding - FW) / stride)
                if out_h * out_w <= 0:
                    print("ERROR : Conv"+str(nums["Conv"]+1)+" : output size is zero or minus !")
                    struct.write("ERROR : Pool"+str(nums["Pool"]+1)+" : output size is zero or minus !")
                    error = True
                    struct.close()
                    sys.exit(0)
                input_size = [out_w, out_h]

                network.append("Conv"+str(nums["Conv"]+1))
                nums["Conv"] += 1
                args["Conv"+str(nums["Conv"])] = (input_num, output_size, fs, stride, padding)
                input_num = output_size
                print("    *output_size = "+str(input_size))
                struct.write("    *output_size = "+str(input_size)+"\n")
                print("  Conv"+str(nums["Conv"])+"Added!")
                print()
                struct.write("\n")

            elif "p" in layer:
                print("  Pool"+str(nums["Pool"]+1)+" :")
                struct.write("Pool"+str(nums["Pool"]+1)+" :\n")
                fs = 2
                if len(n)>=2:
                    fs = n[1]
                    struct.write("    filter_size : "+str(fs)+"\n")
                padding = 0
                if len(n)>=3:
                    padding = n[2]
                    struct.write("    padding : "+str(padding)+"\n")
                H,W = input_size[0], input_size[1]
                out_h = int(1 + (H - fs) / fs)
                out_w = int(1 + (W - fs) / fs)
                if out_h * out_w <= 0:
                    print("Pool"+str(nums["Pool"]+1)+" : output size is zero or minus !")
                    struct.write("ERROR : Pool"+str(nums["Pool"]+1)+" : output size is zero or minus !")
                    error = True
                    struct.close()
                    sys.exit(0)
                input_size = [out_w, out_h]
                network.append("Pool"+str(nums["Pool"]+1))
                nums["Pool"] += 1
                args["Pool"+str(nums["Pool"])] = (fs, padding)
                print("    *output_size = "+str(input_size))
                struct.write("    *output_size = "+str(input_size)+"\n")
                print("  Pool"+str(nums["Pool"])+"Added!")
                print()
                struct.write("\n")

            elif "a" in layer:
                if len(n)<2:
                    print("Affine Layer needs \"1\" Arguments at Least")
                    sys.exit(0)
                print("  Affine"+str(nums["Affine"]+1)+" :")
                struct.write("Affine"+str(nums["Affine"]+1)+" :\n")
                network.append("Affine"+str(nums["Affine"]+1))
                nums["Affine"] += 1
                output_size = n[1]
                struct.write("    output_size : "+str(output_size)+"\n")
                args["Affine"+str(nums["Affine"])] = (input_num, output_size)
                input_num = output_size
                print("  Affine"+str(nums["Affine"])+"Added!")
                print()
                struct.write("\n")

            elif "s" in layer:
                print("  Sigmoid"+str(nums["Sigmoid"]+1)+" :")
                struct.write("Sigmoid"+str(nums["Sigmoid"]+1)+" :\n")
                network.append("Sigmoid"+str(nums["Sigmoid"]+1))
                nums["Sigmoid"] += 1
                args["Sigmoid"+str(nums["Sigmoid"])] = None
                print("  Sigmoid"+str(nums["Sigmoid"])+"Added!")
                print()
                struct.write("\n")

            elif "r" in layer:
                print("  ReLU"+str(nums["ReLU"]+1)+" :")
                struct.write("ReLU"+str(nums["ReLU"]+1)+" :\n")
                network.append("ReLU"+str(nums["ReLU"]+1))
                nums["ReLU"] += 1
                args["ReLU"+str(nums["ReLU"])] = None
                print("  ReLU"+str(nums["ReLU"])+"Added!")
                print()
                struct.write("\n")

            elif "end" in layer:
                print("  Constracted !")
                struct.write("end")
                break
    else:
        print("Not Found this mode!")
        sys.exit(0)
    struct.close()

    if not error:
        print("\nCoding DNN...", end="", flush=True)
        cf = open("networks\\"+name+".py", "w")
        cf.write("""import sys
import numpy as np
import chainer
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
import wx\n\n""")

        cf.write("class "+name+"(Chain):\n")
        cf.write("""\tdef __init__(self):
\t\tsuper("""+name+""", self).__init__()
\t\tself.output = {}
\t\twith self.init_scope():\n""")
        for l in network:
            if "Conv" in l:
                cf.write("\t\t\tself."+l.lower()+" = L.Convolution2D("+str(args[l][0])+", " +str(args[l][1])+", "+str(args[l][2]))
                if args[l][3] != 1:
                    cf.write(", stride="+str(args[l][3]))
                if args[l][4] != 0:
                    cf.write(", pad="+str(args[l][4]))
                cf.write(")\n")

            if "Affine" in l:
                cf.write("\t\t\tself."+l.lower()+" = L.Linear(" +str(args[l][0])+", "+str(args[l][1])+")\n")

        cf.write("\n\tdef __call__(self, x):\n\t\tout = x\n\t\tself.output = {}\n\n")
        for l in network:
            if "Conv" in l or "Affine" in l:
                cf.write("\t\tout = self."+l.lower()+"(out)\n")

            if "Sigmoid" in l or "ReLU" in l:
                cf.write("\t\tout = F."+l.lower()[:-1]+"(out)\n")

            if "Pool" in l:
                cf.write("\t\tout =  F.max_pooling_2d(out, "+str(args[l][0]))
                if args[l][1] != 0:
                    cf.write(", pad="+str(args[l][1]))
                cf.write(")\n")

            cf.write("\t\tself.output[\""+l+"\"] = out.array\n")
        cf.write("\t\treturn out\n")

        cf.write("\n\tdef get_output(self):\n\t\treturn self.output\n")

        cf.close()
        print("finished", flush=True)

    app = wx.App()
    dialog = wx.DirDialog(None, u'コピー先のフォルダを選択してください',path)
    dialog.ShowModal()
    path = dialog.GetPath()

    shutil.copy("networks\\"+name+".py", path)
