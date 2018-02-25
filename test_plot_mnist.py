import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

y = mnist.train.labels[:10, :]
x = mnist.train.images[:10, :]


def display_digit(index):
    label = y[index].argmax(axis=1)
    image = x[index].reshape([28,28])
    plt.title('Example: {}  Label: {}'.format(index, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()


for i in range(10):
    display_digit(i)

