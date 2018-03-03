import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from Util.models.tf_ff_nn import NeuralNetwork

mnist = input_data.read_data_sets('../../MNIST_data/', one_hot=True)

network = NeuralNetwork([784, 333, 333, 333, 10], cost_func=tf.nn.softmax_cross_entropy_with_logits_v2,
                        activation_func=tf.nn.relu,
                        train_func=tf.train.AdamOptimizer(1e-4).minimize)

with tf.name_scope('Saver'):
    saver = tf.train.Saver()

with tf.Session(
    config=tf.ConfigProto(allow_soft_placement=True)
) as sess:
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter('../../summaries/feedforward/test', sess.graph)
    tf.global_variables_initializer().run()

    epochs = 10
    batch_size = 100
    # examples_per_batch = int(mnist.train.num_examples / batch_size)
    examples_per_batch = 200
    global_step = 0

    cost = 0
    c = 0
    for e in range(epochs):
        for i in range(examples_per_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # summary, c, __ = sess.run([merged]+network.train(), feed_dict=network.get_feed_dict(batch_xs, batch_ys))
            # train_writer.add_summary(summary, global_step + i)
            c, __ = sess.run(network.train(), feed_dict=network.get_feed_dict(batch_xs, batch_ys))

            global_step += 1
            cost += c

        summary, accuracy = sess.run([merged]+network.get_accuracy(), feed_dict=network.get_feed_dict(mnist.test.images, mnist.test.labels))
        test_writer.add_summary(summary, global_step)
        print("Passed epoch {} with total cost {} and instantaneous cost {}".format(e, cost, c))
        print("Accuracy: {0:.2f}%".format(accuracy*100))

    save_path = saver.save(sess, '../../saved_models/mnist/model.ckpt')
    print('Model saved: {}'.format(save_path))


