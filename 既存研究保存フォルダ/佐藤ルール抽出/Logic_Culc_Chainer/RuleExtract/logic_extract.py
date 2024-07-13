'''
Created on 2019/07/01

@author: ail
'''

import threading
import numpy as np
import cupy as cp
import os
import time
import glob
import re
from multiprocessing import Pool

class Logic:
    def __init__(self, model, path, input_part, output_part, input_type_l):
        #super(Logic, self).__init__()
        self.model = model
        self.input_part = input_part
        self.output_part = output_part
        self.it_l = input_type_l
        self.path = path+"\\logic"
        self.path_f = self.path+"\\logic_formula"
        self.path_r = self.path+"\\process_result"
        self.logics = []
        self.f_to_f_MS = []
        self.finished = False
        self.data_size = 0
        self.sizes = []
        if self.model != None:
            self.sizes = self.model.getOutput_size()
            for i in range(self.sizes[self.output_part][0]):
                self.logics.append({})
                self.f_to_f_MS.append({})

    def extract(self, out):
        #print(out[self.input_part])
        #論理式抽出
        op = cp.asnumpy(out[self.input_part]).flatten()  # @UndefinedVariable
        op = -2*(op**2)+3*op
        op = op.astype(int)

        oo = np.asarray(out[self.output_part])  # @UndefinedVariable
        oo = -2*(oo**2)+3*oo
        oo = oo.astype(int)

        #print(str(op)+"   "+str(oo))

        r = []
        for u in range(len(oo)):
            f = ""
            f_MS = ""
            if oo[u] == 1:
                for i in range(len(op)):
                    if op[i] == 1:
                        if f == "":
                            f += "P_x"+str(i)
                            f_MS += "x_"+str(i)
                        else:
                            f += "∧P_x"+str(i)
                            f_MS += "∧x_"+str(i)
                    else:
                        if f == "":
                            f += "N_x"+str(i)
                            f_MS += "(x_"+str(i)+")\\bar"
                        else:
                            f += "∧N_x"+str(i)
                            f_MS += "∧(x_"+str(i)+")\\bar"

                r.append([u,f,f_MS])

        return r
        """
        p = Pool(int(os.cpu_count()/2-1))
        #args = itrt.izip(itrt.repeat(self), itrt.repeat('extract'), initial_num_list)
        result_list = p.map(self.extract, initial_num_list)
        p.close()
        """
    def add_Logics(self, i1):
        for i2 in i1:
            for i in i2:
                if len(i) != 0:
                    u = i[0]
                    f = i[1]
                    f_MS = i[2]
                    if not f in self.logics[u].keys():
                        self.logics[u][f] = 1
                        self.f_to_f_MS[u][f] = f_MS
                    else:
                        self.logics[u][f] += 1

    def load_Logics(self, path):
        for f in glob.glob(path+"\\logic\\logic_formula\\*"):
            self.logics.append({})
            text = f.split("\\")[-1]
            unit = int(text.replace("unit-","").replace(".txt",""))
            file = open(f,"r")
            for s in file.read().split("\n"):
                if s != "" :
                    if s[0] == "P" or s[0] == "N":
                        line = s.split(" ")
                        self.logics[unit][line[0]] = int(line[1])
            file.close()

    def sort_output(self,path):
        sort = []
        for u in self.logics:
            sort.append(dict(sorted(u.items(), key=lambda x: x[1], reverse=True)))
        self.logics = sort

        for u in range(len(sort)):
            wf =  open(path+"\\unit-"+str(u)+".txt","w")
            wf.write("================================\n")
            wf.write("    素子"+str(u)+"\n")
            wf.write("================================\n\n")
            wf.write("論理式 出現回数\n")
            for k,v in sort[u].items():
                wf.write(k+" "+str(v)+"\n")
            wf.write("\n")
            wf.write("論理式(Offine数式対応) 出現回数\n")
            for k,v in sort[u].items():
                wf.write(self.f_to_f_MS[u][k]+" "+str(v)+"\n")
            wf.close()

    def test(self, n):
        out = self.model.predict_Test(n)
            #print(type(out["Pool"].array))
        op = cp.asnumpy(out[self.input_part]).flatten()  # @UndefinedVariable
        op = -2*(op**2)+3*op
        op = op.astype(int)

        f = ""
        for i in range(len(op)):
            if op[i] == 1:
                if f == "":
                    f += "P_x"+str(i)
                else:
                    f += "∧P_x"+str(i)
            else:
                if f == "":
                    f += "N_x"+str(i)
                else:
                    f += "∧N_x"+str(i)
        output = []
        for u in self.logics:
            output.append(int(f in u.keys()))
        ans = self.model.getTestAnswer(n)
        s = "logic_output = "+str(output)+", answer = "+str(ans)
        if output[ans] == 1:
            if np.sum(np.array(output)) == 1:
                s += "  ○\n"
            else:
                max = 0
                maxarg = -1
                index = [i for i, x in enumerate(output) if x == 1]
                s += " ※\n"
                for i in index:
                    value = self.logics[i][f]
                    if max < value:
                        max = value
                        maxarg = i
                    s += "   素子"+str(i)+"→"+str(value)+"回\n"
                if maxarg == ans:
                    s+= "  logic_output = "+str(maxarg)
                    s += "   → ○\n"
                else:
                    s += "  logic_output = "+str(maxarg)
                    s += "   → ×\n"
        else:
            s += "  ×\n"
        return s


    def end(self, num, s):
        wp = open(self.path_r +"\\test_processs.txt", "w")
        count = 0
        count_b = 0
        for i in s:
            if "○" in i:
                count += 1
            if "※" in i:
                count_b += 1
            wp.write(i)
        wp.write("\n\n====================================================\n")
        wp.write("データ数 = "+str(num)+", 正答数 = "+str(count)+"\n")
        wp.write("        →正答率 = "+str(count*100/num)+"％\n")
        wp.write("論理式重複数 = "+str(count_b)+"\n")
        wp.write("        →正答率(重複=×) = "+str((count-count_b)*100/num)+"％\n")
        wp.close()

        self.finished = True
        return

    def get_Logics(self):
        return self.logics

    def isFinished(self):
        return self.finished