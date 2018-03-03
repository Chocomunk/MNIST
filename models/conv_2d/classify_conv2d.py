import tensorflow as tf
from Util.models.model_conv2d import build_model

x, y_, logits, train_step, saver, accuracy, cross_entropy, keep_prob, beta = build_model()

labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
          'eight', 'nine']

with tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True)
) as sess:
    writer = tf.summary.FileWriter('../../summaries/test/conv2d', sess.graph)

    saver.restore(sess, '../../saved_models/conv2d/model.ckpt')
    print('Model restored')

    for label in labels:
        file_contents = tf.read_file('../../test_images/{}.png'.format(label))
        image_raw = tf.image.decode_png(file_contents, channels=1)
        image_tensor = tf.reshape(image_raw, shape=[-1, 784])

        image = sess.run(image_tensor)
        label_out_op = tf.argmax(logits, axis=1)

        output, label_out = sess.run([logits, label_out_op], feed_dict={x: image, keep_prob: 1.0})
        print("{}: {}, {}".format(label, label_out, output))

