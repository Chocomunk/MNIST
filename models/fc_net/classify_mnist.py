import tensorflow as tf
import numpy as np
from tf_ff_nn import NeuralNetwork

network = NeuralNetwork([784, 333, 333, 333, 10], cost_func=tf.nn.softmax_cross_entropy_with_logits,
                        activation_func=tf.nn.relu,
                        train_func=tf.train.AdamOptimizer().minimize,
                        write_summaries=False)

with tf.name_scope('Saver'):
    saver = tf.train.Saver()

label = 'four'
file_contents = tf.read_file('../../test_images/{}.png'.format(label))
image_raw = tf.image.decode_png(file_contents, channels=1)
# image_tensor = tf.image.resize_images(image_raw, size=[1, 784])
image_tensor = tf.reshape(image_raw, shape=[-1, 784])
logits = tf.nn.softmax(network.get_output_layer(), name='Softmax_Logits')

with tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True)
) as sess:
    # merged = tf.summary.merge_all()
    # writer = tf.summary.FileWriter('summaries/test/mnist', sess.graph)

    image = sess.run(image_tensor)
    saver.restore(sess, '../../saved_models/mnist/model.ckpt')
    print('Model restored')

    # output = sess.run(logits, feed_dict={network.get_train_data()[0]: image})
    output = sess.run(network.get_output_layer(), feed_dict={network.get_train_data()[0]: image})
    # writer.add_summary(summary, 0)
    print(output)

