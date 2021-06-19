from mnist import MNIST
import numpy as np 
import matplotlib.pyplot as plt
import imageio as smisc


data = MNIST('mlengine/mnist_data')

images, labels = data.load_testing()


images = np.asarray(images)
labels = np.asarray(labels)

for i in range(5):
    im = images[i].reshape(28,28)
    smisc.imsave('mlengine/mnist_data{}.png'.format(i), im)
