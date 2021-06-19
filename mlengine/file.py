import os
from PIL import Image
import numpy as np


# all_img_files = [f for f in os.listdir('D:\DATA FOR MACHINE LEARNING PROJECT\MNIST_model_project\mlengine\pics prediction') if f.endswitch('.jpg')]
path = 'D:\DATA FOR MACHINE LEARNING PROJECT\MNIST_model_project\mlengine\pics prediction'
l = os.listdir(path)
# print(l)
img = []
for i in l:
    image = Image.open(f'D:\DATA FOR MACHINE LEARNING PROJECT\MNIST_model_project\mlengine\pics prediction\{i}')
    img.append(np.asarray(image))

print(img)

