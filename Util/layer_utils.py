import tensorflow as tf

epsilon = 1e-7


def weight(shape):
    with tf.name_scope('weight'):
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
    return w


def bias(shape):
    with tf.name_scope('bias'):
        b = tf.Variable(tf.constant(0.1, shape=shape), name='bias')
    return b


def conv2d(data, weight_filter):
    return tf.nn.conv2d(data, weight_filter, strides=[1, 1, 1, 1],
                        padding='SAME', name='convolution')


def max_pool_2x2(data):
    return tf.nn.max_pool(data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='max_pooling')


def relu_wrapper(inputs):
    return tf.nn.relu(inputs, name='activation')


def batch_norm(inputs, is_training, is_convolutional, decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        axes = None
        if is_convolutional:
            axes = [i for i in range(len(inputs.get_shape())-1)]
        else:
            axes = [0]
        batch_mean, batch_var = tf.nn.moments(inputs, axes=axes)
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1-decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1-decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta,
                                             scale, epsilon, name='batch_normalization')
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta,
                                         scale, epsilon, name='batch_normalization')
