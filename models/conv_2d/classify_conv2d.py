import tensorflow as tf

def weight(shape, isFilter):
    with tf.name_scope('weight'):
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        # if isFilter:
        #     with tf.device('/cpu:0'):
        #         images = tf.transpose(w, [3, 2, 1, 0])
        #         images = tf.split(images, shape[3], 0)
        #         for i in range(shape[3]):
        #             image = tf.reshape(images[i],
        #                                [1, shape[1]*shape[2], shape[0], 1])
        #             tf.summary.image('filter_{}'.format(i), image)
        #             # images = list(tf.split(images, shape[2], 3))
        #             # images = tf.stack(images)
        #             # images = tf.reshape(images,
        #             #                     [1, shape[1]*shape[2]*shape[3], shape[0], 1])
        #             # tf.summary.image('filter', images)
    return w


def bias(shape):
    with tf.name_scope('bias'):
        b = tf.Variable(tf.constant(0.1, shape=shape), name='bias')
        # tf.summary.scalar('biases', b)
    return b


def conv2d(data, weight_filter):
    return tf.nn.conv2d(data, weight_filter, strides=[1, 1, 1, 1],
                        padding='SAME', name='convolution')


def max_pool_2x2(data):
    return tf.nn.max_pool(data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='max_pooling')


# with tf.device('/cpu:0'):
with tf.name_scope('Setup'):
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='input')
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # x_image = tf.reshape(x, [-1, 28, 28, 1])

# with tf.device('/gpu:0'):
with tf.name_scope('Model'):
    with tf.name_scope('Conv_1'):
        w_conv1 = weight([5, 5, 1, 32], True)
        b_conv1 = bias([32])
        h_conv1 = tf.nn.relu(tf.add(conv2d(x, w_conv1), b_conv1), name='activation')
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('Conv_2'):
        w_conv2 = weight([5, 5, 32, 64], True)
        b_conv2 = bias([64])
        h_conv2 = tf.nn.relu(tf.add(conv2d(h_pool1, w_conv2), b_conv2), name='activation')
        h_pool2 = max_pool_2x2(h_conv2)

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
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits
                                                    , name='cross_entropy'))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, name='Adam')

# with tf.device('/cpu:0'):
with tf.name_scope('Evaluation'):
    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    with tf.name_scope('Saver'):
        saver = tf.train.Saver()

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
        # image_tensor = tf.image.resize_images(image_raw, size=[-1, 784])
        image_tensor = tf.reshape(image_raw, shape=[-1, 28, 28, 1])

        with tf.name_scope(label):
            tf.summary.image('input_img', tf.reshape(x, [1,28,28,1]))
            tf.summary.image('input_val', tf.reshape(x, [1,28,28,1]))
            tf.summary.image('Layer_1/Conv', tf.reshape(h_conv1, [1, 28*32, 28, 1]))
            tf.summary.image('Layer_1/Pool;', tf.reshape(h_pool1, [1, 14*32, 14, 1]))
            tf.summary.image('Layer_2/Conv', tf.reshape(h_conv2, [1,14*64,14,1]))
            tf.summary.image('Layer_2/Pool', tf.reshape(h_pool2, [1, 7*64, 7, 1]))
            # tf.summary.image('logits', logits)

        merged = tf.summary.merge_all()
        image = sess.run(image_tensor)

        output, summary = sess.run([logits, merged], feed_dict={x: image, keep_prob: 1.0})
        writer.add_summary(summary, 0)
        print("{}: {}".format(label, output))
