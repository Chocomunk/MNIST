import tensorflow as tf
import numpy as np

from Util.model_conv2d_batch_norm import build_model

x, y_, logits, train_step, saver, accuracy, cross_entropy, keep_prob, beta = build_model(False)


labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
          'eight', 'nine']

with tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True)
) as sess:
    writer = tf.summary.FileWriter('../../summaries/test/conv2d_batch_norm', sess.graph)

    saver.restore(sess, '../../saved_models/conv2d_batch_norm/model.ckpt')
    print('Model restored')

    for label in labels:
        file_contents = tf.read_file('../../test_images/{}.png'.format(label))
        image_raw = tf.image.decode_png(file_contents, channels=1)
        image_tensor = tf.reshape(image_raw, shape=[-1, 784])

        merged = tf.summary.merge_all()
        image = sess.run(image_tensor)

        # output, summary = sess.run([logits, merged], feed_dict={x: image, keep_prob: 1.0})
        output = sess.run([logits], feed_dict={x: image, keep_prob: 1.0})
        # label_out_op = tf.argmax(output, axis=[0,1,2])
        # label_out = sess.run(label_out_op)
        numpy_out = np.asarray(output)
        label_out = numpy_out.argmax()
        # writer.add_summary(summary, 0)
        print("{}: {}, {}".format(label, label_out, numpy_out))
