'''
Created on 2018/07/06

@author: ail
'''

class bool_culc:

    def __init__(self, param, flag):
        self.prelogic = param
        self.logic = param.replace('_x','').split(' AND ')
        self.Flag = flag

    def changeLogic(self, newLogic):
        self.logic = newLogic.replace('_x','').split(' AND ')

    def culc(self, input):
        result = 1
        if self.logic[0] == "null":
            return 0
        for l in self.logic:
            num = int(l[1:])
            if(l[0]=='P'):
                result *= input[num]
            else:
                result *= abs(input[num]-1)
        return result

    def getLogic(self):
        return self.prelogic