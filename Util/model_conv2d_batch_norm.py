import tensorflow as tf
from Util.layer_utils import *

def build_model(is_training):
    with tf.name_scope('Setup'):
        x = tf.placeholder(tf.float32, shape=[None, 784], name='input')
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('Model'):
        keep_prob = tf.placeholder(tf.float32)

        with tf.name_scope('Conv_1'):
            w_conv1 = weight([5, 5, 1, 32])
            b_conv1 = bias([32])
            act_conv1 = tf.add(conv2d(x_image, w_conv1), b_conv1)
            bn_conv1 = batch_norm(act_conv1, is_training, True)
            h_conv1 = relu_wrapper(bn_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

        with tf.name_scope('Dropout_1'):
            h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob, name='dropout')

        with tf.name_scope('Conv_2'):
            w_conv2 = weight([5, 5, 32, 64])
            b_conv2 = bias([64])
            act_conv2 = tf.add(conv2d(h_pool1_drop, w_conv2), b_conv2)
            bn_conv2 = batch_norm(act_conv2, is_training, True)
            h_conv2 = relu_wrapper(bn_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

        with tf.name_scope('Dropout_2'):
            h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob, name='dropout')

        with tf.name_scope('Fully_Connected'):
            w_fc1 = weight([7 * 7 * 64, 1024])
            b_fc1 = bias([1024])
            h_pool2_flat = tf.reshape(h_pool2_drop, [-1, 7 * 7 * 64])
            act_dense1 = tf.matmul(h_pool2_flat, w_fc1) + b_fc1
            bn_dense1 = batch_norm(act_dense1, is_training, False)
            h_fc1 = tf.nn.relu(bn_dense1, name='fully_connected')

        with tf.name_scope('Dropout_3'):
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='dropout')

        with tf.name_scope('Logits'):
            w_fc2 = weight([1024, 10])
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
