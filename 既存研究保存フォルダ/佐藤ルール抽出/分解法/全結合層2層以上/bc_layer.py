'''
Created on 2018/07/18

@author: ail
'''

from bool_culc import bool_culc as bc  # @UnresolvedImport

class bc_layer:
    '''
    classdocs
    '''

    def __init__(self):
        self.layer = []

    def output(self, inputs):
        ans = []
        for i in self.layer:
            o = i.culc(inputs)
            ans.append(o)
        return ans

    def addItem(self, item, flag):
        self.layer.append(bc(param = item, flag = flag))

    def getNum(self):
        return len(self.layer)

    def getItem(self):
        ans = []
        for i in self.layer:
            ans.append(i.getLogic())
        return ans