import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def weight(shape, is_filter):
    with tf.name_scope('weight'):
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        if is_filter:
            with tf.device('/cpu:0'):
                images = tf.transpose(w, [3, 2, 1, 0])
                images = tf.split(images, shape[3], 0)
                for i in range(shape[3]):
                    image = tf.reshape(images[i],
                                       [1, shape[1]*shape[2], shape[0], 1])
                    tf.summary.image('filter_{}'.format(i), image)
    return w


def bias(shape):
    with tf.name_scope('bias'):
        b = tf.Variable(tf.constant(0.1, shape=shape), name='bias')
        # tf.summary.scalar('biases', b)
    return b


def conv2d(data, weight_filter):
    return tf.nn.conv2d(data, weight_filter, strides=[1, 1, 1, 1],
                        padding='SAME', name='convolution')


def max_pool_2x2(data):
    return tf.nn.max_pool(data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='max_pooling')


mnist = input_data.read_data_sets('../../MNIST_data/', one_hot=True)

# with tf.device('/gpu:0'):
with tf.name_scope('Setup'):
    x = tf.placeholder(tf.float32, shape=[None, 784], name='input')
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

# with tf.device('/gpu:0'):
with tf.name_scope('Model'):
    with tf.name_scope('Conv_1'):
        w_conv1 = weight([5, 5, 1, 32], True)
        b_conv1 = bias([32])
        h_conv1 = tf.nn.relu(tf.add(conv2d(x_image, w_conv1), b_conv1), name='activation')
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('Conv_2'):
        w_conv2 = weight([5, 5, 32, 64], True)
        b_conv2 = bias([64])
        h_conv2 = tf.nn.relu(tf.add(conv2d(h_pool1, w_conv2), b_conv2), name='activation')
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('Fully_Connected'):
        w_fc1 = weight([7 * 7 * 64, 1024], False)
        b_fc1 = bias([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1, name='fully_connected')

    with tf.name_scope('Dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='dropout')

    with tf.name_scope('Logits'):
        w_fc2 = weight([1024, 10], False)
        b_fc2 = bias([10])
        y_conv = tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2, name='logit_conv')
        logits = tf.nn.softmax(y_conv, name='Softmax_Logits')

    with tf.name_scope('Train'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv
                                                    , name='cross_entropy'))
        regularizers = (tf.nn.l2_loss(w_conv1) + tf.nn.l2_loss(w_conv2) +
                        tf.nn.l2_loss(w_fc1) + tf.nn.l2_loss(w_fc2))
        beta = tf.placeholder(tf.float32)
        loss = tf.reduce_mean(cross_entropy + beta * regularizers)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, name='Adam')

# with tf.device('/gpu:0'):
with tf.name_scope('Evaluation'):
    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        tf.summary.scalar("accuracy", accuracy)

    with tf.name_scope('Saver'):
        saver = tf.train.Saver()

with tf.Session(
        # config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config=tf.ConfigProto(allow_soft_placement=True)
) as sess:
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter('../../summaries/conv2d/test', sess.graph)
    tf.global_variables_initializer().run()

    epochs = 50
    batch_size = 100
    # examples_per_batch = int(mnist.train.num_examples / batch_size)
    examples_per_batch = 200

    cost = 0
    cost_i = 0
    for e in range(epochs):
        for i in range(examples_per_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            cost_i, __ = sess.run([cross_entropy, train_step],
                                  feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5, beta: 0.01})
            cost += cost_i

        summary, accuracy_val = sess.run([merged, accuracy],
                                         feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        test_writer.add_summary(summary, e * examples_per_batch)
        print("Passed epoch {} with total cost {} and instantaneous cost {}".format(e, cost, cost_i))
        print("Accuracy: {0:.2f}%".format(accuracy_val * 100))

    save_path = saver.save(sess, '../../saved_models/conv2d/model.ckpt')
    print("Model saved: {}".format(save_path))
