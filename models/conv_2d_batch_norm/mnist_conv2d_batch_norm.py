import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from Util.models.model_conv2d_batch_norm import build_model

mnist = input_data.read_data_sets("../../MNIST_data/", one_hot=True)

x, y_, logits, train_step, saver, accuracy, cross_entropy, keep_prob, beta = build_model(True)


with tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True)
) as sess:
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter('../../summaries/conv2d_batch_norm/train', sess.graph)
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
                                  feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5, beta: 0.02})
            cost += cost_i

        summary, accuracy_val = sess.run([merged, accuracy],
                                         feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        test_writer.add_summary(summary, e * examples_per_batch)
        print("Passed epoch {} with total cost {} and instantaneous cost {}".format(e, cost, cost_i))
        print("Accuracy: {0:.2f}%".format(accuracy_val * 100))

    save_path = saver.save(sess, '../../saved_models/conv2d_batch_norm/model.ckpt')
    print("Model saved: {}".format(save_path))
