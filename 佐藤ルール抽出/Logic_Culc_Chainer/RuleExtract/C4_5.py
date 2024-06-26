'''
Created on 2019/11/21

@author: ail
'''
import numpy as np

class C4_5:
    def __init__(self):
        pass

    def generate(self, xy):
        pass

class Node:
    def __init__(self):
        self.rule = None

    def generateNode(self, xy):
        sumx = 0
        sumxy = 0
        sumxy_inv = 0
        for n in xy:
            (x, y) = n
            #print(x,y)
            sumx += x
            sumxy += x*y
            sumxy_inv += x*(1-y)
        px = sumx / len(xy)
        py_x = sumxy / len(xy)
        py_x_inv = sumxy_inv / len(xy)
        #print(py_x)
        Hc = -px*np.log2(px+1e-10) - (1-px)*np.log2(1-px+1e-10)
        Hc_a = px*(-py_x*np.log2(py_x+1e-10) - (1-py_x)*np.log2(1-py_x+1e-10))\
            + (1-px)*(-py_x_inv*np.log2(py_x_inv+1e-10) - (1-py_x_inv)*np.log2(1-py_x_inv+1e-10))
        Ic = Hc - Hc_a
        arg = np.argmax(Ic)
        self.rule = (arg, 0.5)
        return (arg, 0.5)

    def set_node(self, rule):
        self.rule = rule

    def get_node(self):
        return self.rule