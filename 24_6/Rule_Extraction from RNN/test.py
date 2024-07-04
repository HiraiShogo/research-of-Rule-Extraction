import numpy as np
import sys


a = np.ones((2)) * 2
b = np.ones((3)) * 3
c = np.concatenate((a, b))

print(c.shape)

