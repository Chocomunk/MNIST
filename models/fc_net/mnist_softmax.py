import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../../MNIST_data/', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
w = tf.Variable(tf.truncated_normal([784, 10], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[10]))

y = tf.matmul(x, w) + b
# y_e = tf.nn.softmax(y)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    # print(sess.run(w))
    # print(sess.run(b))

    # saver = tf.train.Saver([w,b])
    # saver.save(sess, '../MNIST/Saves/Checkpoints/lol.chk')

    # im = Image.open('zero.png')
    # im_g  = im.convert('L')
    # print(np.array(im).shape)
    # Image.fromarray(np.array(im_g).transpose().reshape(1,784).transpose().reshape(28,28), mode='L').show()
    # print("\n\nEval with zero png:\n{0}".format(sess.run(y_e, feed_dict={x:np.array(im_g).transpose().reshape(1,784),
    # 																	 w:sess.run(w),
    # 																	 b:sess.run(b)})))
