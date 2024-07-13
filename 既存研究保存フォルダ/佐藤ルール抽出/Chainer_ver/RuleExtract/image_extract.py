'''
Created on 2019/07/02

@author: ail
'''

import os
import cupy as cp
import numpy as np
import time
#from multiprocessing import Pool
from PIL import Image as PImage
import threading

class Image_save:
    def __init__(self, model, path, outpart):
        self.model = model
        self.finished = False
        self.size = int(cp.prod(cp.asarray(self.model.getOutput_size()[outpart])))

    def save_ImageI(self, Units, noised, path_t, path_noise):
        for i in range(Units):
            """
            a = np.uint8(cp.asnumpy(self.model.getTrain(i)) * 255)
            img = PImage.fromarray(a)
            img.save(path_t+"\\unit-"+str(i)+'.jpg')
            """
            b = np.uint8(self.normalize(cp.asnumpy(noised[i][0,0])) * 255)
            img2 = PImage.fromarray(b)
            img2.save(path_noise+"\\unit-"+str(i)+'.jpg')

    def save_ImageG(self, grads, path_n, path_s, path_z, path_sz):
        for i in range(grads.shape[0]):
            a1 = self.normalize(grads[i])
            a2 = np.uint8(cp.asnumpy(a1) * 255)
            img = PImage.fromarray(a2)

            img.save(path_n+"\\unit-"+str(i)+'.jpg')
            img.resize((168,168)).save(path_z+"\\unit-"+str(i)+'.jpg')

            a3 = 3*a1-2*a1**2
            a3 = np.uint8(cp.asnumpy(a3).astype(np.int) * 255)
            img2 = PImage.fromarray(a3)
            img2.save(path_s+"\\unit-"+str(i)+'.jpg')
            img2.resize((168,168)).save(path_sz+"\\unit-"+str(i)+'.jpg')

    def normalize(self, i):
        max = np.max(i.flatten())
        min = np.min(i.flatten())
        return (i-min)/(max-min)

    def isFinished(self):
        return self.finished