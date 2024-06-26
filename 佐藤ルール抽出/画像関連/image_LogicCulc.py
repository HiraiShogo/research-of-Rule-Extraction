'''
Created on 2018/10/30

@author: ail
'''
import glob
from PIL import Image
import numpy as np
import os
import wx
import datetime
import subprocess

pathI = "image\\2018-10-25 13_51_03_727630"
pathL = "logic\\ruleAll_delite2"

if __name__ == '__main__':
    app = wx.App()
    pathF = "C:\\pleiades\\workspace\\Logic_Culc\\culc\\"
    dialog = wx.DirDialog(None, u'フィルタ画像のフォルダを選択してください', pathF)
    dialog.ShowModal()

    pathI = dialog.GetPath()
    print("ImageFolder: "+pathI)

    dialog = wx.DirDialog(None, u'論理式のフォルダを選択してください', pathF)
    dialog.ShowModal()
    pathL = dialog.GetPath()
    print("LogicFolder: "+pathL)

    image = []
    for i in glob.glob(pathI+'\\result\\*'):
        image.append(np.array(Image.open(i)))
    image = np.array(image)
    print(image.shape)

    image_NOT = 255-image
    #print(image_NOT)

    logic = []
    arynum = 0
    for i in glob.glob(pathL + '\\test_result\\*'):
        logic.append([])
        with open(i) as f:
            l = f.read()
        for j in l.split('\n'):
            if len(j) == 0:
                break
            else:
                logic[arynum].append(j.split(',')[0])
        arynum += 1

    todaydetail = input(' name > ')
    if len(todaydetail) == 0:
        todaydetail = str(datetime.datetime.today()).replace(".", "_").replace(":","_")
    for i in range(len(logic)):
        os.makedirs("Logic_Image\\"+str(todaydetail)+"\\out"+str(i))
        pic = np.zeros(image[0].shape)
        for j in logic[i]:
            s = 0
            output = []
            for k in j.split(" AND "):
                tf = k[0]
                num = int(k[3])
                if s == 0:
                    s += 1
                    if tf == "P":
                        output = np.array(image[num])
                    elif tf == "N":
                        output = np.array(image_NOT[num])
                else:
                    for h in range(len(output)):
                        for w in range(len(output[h])):
                            if tf == "P":
                                if output[h,w]>image[num,h,w]:
                                    output[h,w] =image[num,h,w]
                            elif tf == "N":
                                if output[h,w]>image_NOT[num,h,w]:
                                    output[h,w] = image[num,h,w]
            print(output)
            pic = np.maximum(output,pic)
            Image.fromarray(output).save("Logic_Image\\" +str(todaydetail)+"\\out"+str(i)+"\\"+str(j)+".jpg")
        #print(pic)
        Image.fromarray(pic).convert("L").save("Logic_Image\\" +str(todaydetail)+"\\out"+str(i)+"_OR.jpg")

    cmd = 'Logic_Image\\'+str(todaydetail)
    subprocess.Popen(['explorer',cmd])