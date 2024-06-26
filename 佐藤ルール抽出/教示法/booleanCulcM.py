import numpy as np


class booleanCulcM:

    def __init__(self, logic):
        self.logic = logic

    def culc(self, inputs):
        for l in self.logic:
            o = 100
            ans = 0
            for pn in l.split(' AND '):
                i = inputs[int(pn[-1])]
                if pn[0] == "P":
                    o = self.minimum(o,i)
                elif pn[0] == "N":
                    o= self.minimum(o, 1-i)
            ans = self.maximum(ans, o)
        return 0

    def minimum(self,a,b):
        if a>b:
            return b
        else:
            return a
    def maximum(self,a,b):
        if a>b:
            return a
        else:
            return b
