
class booleanCulc:

    def __init__(self, logic):
        self.logic = logic

    def culc(self, inputs):
        for l in self.logic:
            num = 1
            for pn in l.split(' AND '):
                if pn[0] == "P" and inputs[int(pn[-1])] == 0:
                    num = 0
                    break
                elif pn[0] == "N" and inputs[int(pn[-1])] == 1:
                    num = 0
                    break
            if num==1:
                return 1
        return 0
