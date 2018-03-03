import tensorflow as tf
from Util.layer_wrappers import *


def residual_block(inputs, filter_dim=5, filter_depths=(32, 64)):
    with tf.name_scope('aggregate_block'):
        w_conv1 = weight([filter_dim, filter_dim, 1, filter_depths[0]], True)
        b_conv1 = bias([filter_depths[0]])
        h_conv1 = conv2d(inputs, w_conv1) + b_conv1
        act_1 = relu_wrapper(h_conv1)
        pool_1 = max_pool(act_1)
        w_conv2 = weight([filter_dim, filter_dim, *filter_depths], True)
        b_conv2 = bias([filter_depths[1]])
        h_conv2 = conv2d(pool_1, w_conv2) + b_conv2
        pool_2 = max_pool(h_conv2)
        input_pool = max_pool(inputs, dim=4)
        return relu_wrapper(pool_2+input_pool), w_conv1, w_conv2
