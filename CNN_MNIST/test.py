import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('dataset/both.png').convert('RGB')
plt.imshow(img)
plt.show()