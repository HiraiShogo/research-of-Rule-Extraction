'''
Created on 2018/06/01

@author: ail
'''

import glob
from convolutional_network import SimpleConvNet# @UnresolvedImport
from dataset.mnist import load_mnist  # @UnresolvedImport
import datetime
import time
import os
import subprocess

path = 'logic\\2018-12-20 10_50_35_004495'
Units = 10000
Flag = True #True:Notあり False:Notなし

if __name__ == '__main__':
    network = SimpleConvNet(input_dim=(1,28,28),
                        conv_param = {'filter_num': 10, 'filter_size': 8, 'pad': 0, 'stride': 1},
                        conv2_param = {'filter2_num': 10, 'filter2_size': 7, 'pad2': 0, 'stride2': 1},
                        conv3_param = {'filter3_num': 10, 'filter3_size': 14, 'pad3': 0, 'stride3': 1},
                        hidden_size=10, output_size=10, weight_init_std=0.01)
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    dataList = x_test[:Units]

    x1 = 0 #rnd.randint(0,12)
    y1 = 0 #rnd.randint(0,12)

    files = glob.glob(path+'\\*')
    todaydetail = str(datetime.datetime.today()).replace(".", "_").replace(":","_")
    os.makedirs("logic\\"+str(todaydetail))
    best_logic = []
    for i in files:
        best_logic.append("")
    start = time.time()

    number = -1

    for file in files:
        with open(file+"\\logic_output.txt") as f:
            logic = f.read()

        number += 1
        os.makedirs("logic\\"+str(todaydetail)+ "\\" + str(number))
        fw = open(('logic\\'+str(todaydetail) + "\\" + str(number) + "\\result.txt"), "w")
        fw.write("\'Not\' enabled: "+str(Flag)+"\n\n")
        output = 0

        acc = 0
        max = 0
        dim = len(logic.split('\n')[0].split(' AND '))
        for i in logic.split('\n'):
            #if(len(i.split(' AND ')) > dim):
                #break;
            if(len(i)!=0):
                #print(i)
                fw.write(i+':\n')
                sum = 0
                for u in range(0, Units):
                    result = 1
                    #print("  "+str(u)+"回目")
                    network_output = network.predict(dataList[u].reshape((1,1,28,28)))
                    for j in i.split(' AND '):
                        arg2 = j.replace('_x','')
                        if(network_output[0,int(arg2[1:])-1,x1,y1]>=0.5):#
                            output = 1
                        else :
                            output = 0
                    #print("    "+str(arg2)+", " + str(output)+" -> "+(arg2[0]=='P' and str(output) or str(abs(output-1))))
                        if(arg2[0]=='P'):
                            result *= output
                        else:
                            if(Flag):
                                result *= abs(output-1)
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
                fw.write('acc : '+str(acc)+"\n")
                fw.write("===========================================\n\n")
        print("file"+str(file)[-1]+" finished")

    os.makedirs("logic\\"+str(todaydetail)+ "\\all")
    fw = open(('logic\\'+str(todaydetail) + "\\all\\result.txt"), "w")
    fw.write("\'Not\' enabled: "+str(Flag)+"\n\n")
    sum = 0
    for u in range(0, Units):
        l = 0
        for i in best_logic:
            result = 1
            network_output = network.predict(dataList[u].reshape((1,1,28,28)))#[0,int(arg2[1])-1,x1,y1]
            #print("  "+str(u)+"回目")
            for j in i.split(' AND '):
                arg2 = j.replace('_x','')
                if(network_output[0,int(arg2[1:])-1,x1,y1]>=0.5):#
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

    for i in range(0,len(best_logic)):
        fw.write(str(i)+": "+best_logic[i]+"\n")

    end = time.time()
    print("All finished - "+str(end - start)+"[sec]")
    print(best_logic)

    cmd = 'logic\\'+str(todaydetail)
    subprocess.Popen(['explorer',cmd])
