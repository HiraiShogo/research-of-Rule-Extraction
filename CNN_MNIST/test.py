import os


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

im = Image.open('./pic/doberman.png')
im = im.resize((224,224))
im_list = np.asarray(im)
print(im_list.shape)
plt.imshow(im_list)
plt.show()