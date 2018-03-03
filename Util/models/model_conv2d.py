import tensorflow as tf
from Util.layer_wrappers import *


def build_model():
    with tf.name_scope('Setup'):
        x = tf.placeholder(tf.float32, shape=[None, 784], name='input')
        y_ = tf.placeholder(tf.float32, shape=[None, 10])

        x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('Model'):
        with tf.name_scope('Conv_1'):
            w_conv1 = weight([5, 5, 1, 32], True)
            b_conv1 = bias([32])
            h_conv1 = tf.nn.relu(tf.add(conv2d(x_image, w_conv1), b_conv1), name='activation')
            h_pool1 = max_pool(h_conv1)

        with tf.name_scope('Conv_2'):
            w_conv2 = weight([5, 5, 32, 64], True)
            b_conv2 = bias([64])
            h_conv2 = tf.nn.relu(tf.add(conv2d(h_pool1, w_conv2), b_conv2), name='activation')
            h_pool2 = max_pool(h_conv2)

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

    with tf.name_scope('Evaluation'):
        with tf.name_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
            tf.summary.scalar("accuracy", accuracy)

        with tf.name_scope('Saver'):
            saver = tf.train.Saver()

    return x, y_, logits, train_step, saver, accuracy, cross_entropy, keep_prob, beta

