import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
          'eight', 'nine']

images = {}

for label in labels:
    file_contents = tf.read_file('test_images/{}.png'.format(label))
    image_raw = tf.image.decode_png(file_contents, channels=1)
    images[label] = tf.reshape(image_raw, shape=[-1, 784])


def display_digit(image, label):
    img = image.reshape([28,28])
    plt.title('Label: {}'.format(label))
    plt.imshow(img, cmap=plt.get_cmap('gray_r'))
    plt.show()


with tf.Session() as sess:
    for label in labels:
        image_tensor = sess.run(images[label])
        display_digit(image_tensor, label)