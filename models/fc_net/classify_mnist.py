import tensorflow as tf
from Util.tf_ff_nn import NeuralNetwork

network = NeuralNetwork([784, 333, 333, 333, 10], cost_func=tf.nn.softmax_cross_entropy_with_logits_v2,
                        activation_func=tf.nn.relu,
                        train_func=tf.train.AdamOptimizer().minimize,
                        write_summaries=False)

with tf.name_scope('Saver'):
    saver = tf.train.Saver()

labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
          'eight', 'nine']
logits = tf.nn.softmax(network.get_output_layer(), name='Softmax_Logits')

with tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True)
) as sess:
    writer = tf.summary.FileWriter('../../summaries/test/mnist', sess.graph)

    saver.restore(sess, '../../saved_models/mnist/model.ckpt')
    print('Model restored')

    for label in labels:
        file_contents = tf.read_file('../../test_images/{}.png'.format(label))
        image_raw = tf.image.decode_png(file_contents, channels=1)
        image_tensor = tf.reshape(image_raw, shape=[-1, 784])

        image = sess.run(image_tensor)
        merged = tf.summary.merge_all()

        output = sess.run(logits, feed_dict={network.get_train_data()[0]: image})
        label_out_op = tf.argmax(output)
        label_out = sess.run(label_out_op)
        # writer.add_summary(summary, 0)
        print("{}: {}, {}".format(label, label_out, output))

