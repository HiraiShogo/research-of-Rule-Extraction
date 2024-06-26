'''
Created on 2018/06/19

@author: ail
'''
import re
import glob
import datetime
import os
import subprocess
from numpy.core.tests.test_mem_overlap import xrange

path = "read_result\\"
num = 0
start = 1

files = glob.glob(path+'\\*')
todaydetail = str(datetime.datetime.today()).replace(".", "_").replace(":","_")

def isEnd(use,dim, maxDim): #終了判定
    value = maxDim
    for i in xrange(dim):
        if use[i] != value:
            return False
        value -= 1
    return True

def countUp(use,maxDim): #カウンタのカウントアップ
    if(use[0] != maxDim):
        use[0] += 1
    else: #繰り上げ
        for i in xrange(1,len(use)):
            if use[i] != maxDim - i:
                use[i] += 1
                for j in xrange(1,i+1):
                    use[i-j] = use[i-j+1] + 1
                break
    return use

def isSame(use, weight_num): #肯定と否定検出 True:同じなし False:同じあり
    oks = []
    for i in use:
        for j in oks:
            if(i == j-weight_num):
                return False
        oks.append(i)
    return True

if __name__ == '__main__':

    for file in files:
        os.makedirs(str(todaydetail)+"\\"+str(file.split("\\")[-1].replace(".txt","").replace("inputs[","").replace("]","")))
        fw = open((str(todaydetail) + "\\"+str(file.split("\\")[-1].replace(".txt","").replace("inputs[","").replace("]",""))+"\\logic_output.txt"), "w")
        with open(file) as f: #ファイル読み込み
            ws = f.read()
        print("input["+str(num)+"] : ")
        num +=1
        weights = []
        use = [] #使用番号
        result = []
        for i in ws.split('\n'):
            weights.append(float(i.lstrip()))

        weight_num =len(weights)-1
        maxDim = weight_num*2
        end = maxDim
        #カウンタ初期化
        for i in xrange(0,start):
            use.append(1)
        for i in xrange(1,start):
            use[len(use)-i-1] = use[len(use)-i]+1

        for dim in xrange(1,end+1): #次数ループ
            while True:
                if(isSame(use,weight_num)):
                    sum = weights[weight_num]
                    logic = ""
                    for w in range(1, weight_num+1): #重み番号ループ
                        if(w in use):
                            sum += weights[w]
                        elif(weights[w]<0 and (w+weight_num) not in use):
                            sum += weights[w]

                    if sum >= 0: #sum>=0 → Sigmoid ≒ 1
                        flag = True
                        for o in use: #使用変数ループ
                            if flag: #最初一回目
                                if o<=weight_num:
                                    logic += "P_x"+str(o-1)
                                else:
                                    logic += "N_x"+str(o-weight_num-1)
                                flag = False
                            else:
                                if o<=weight_num:
                                    logic += " AND P_x"+str(o-1)
                                else:
                                    logic += " AND N_x"+str(o-weight_num-1)

                        if len(result)==0:
                            result.append(logic)
                        else: #吸収律
                            flag3 = False
                            for i in result:
                                flag2 = True
                                for j in i.split(' AND '):
                                    flag2 = flag2 and (j in logic)
                                if(flag2):
                                    flag3 = True
                                    break
                            if(not flag3):
                                result.append(logic)
                                print("  "+str(use) + "  " + str(isEnd(use,dim,maxDim)) + "  " +str(sum) +" => "+logic)

                if isEnd(use,dim,maxDim): #カウンタ最大値
                    use.append(1)
                    for i in range(1,len(use)): #カウンタ初期化
                        use[len(use)-i-1] = use[len(use)-i]+1
                    break
                else:
                    use = countUp(use,maxDim)

        for i in result:
            print(i)
            fw.write(i+"\n")
        print()
    print("finished")

    #終了後フォルダを開く
    cmd = str(todaydetail)
    subprocess.Popen(['explorer',cmd])
