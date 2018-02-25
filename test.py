import tensorflow as tf

with tf.device('gpu:0'):
    a = tf.constant([1,2], shape=[1,2])
    b = tf.constant([[2],[3]], shape=[2,1])
    c = tf.matmul(a,b)

with tf.Session(
    config=tf.ConfigProto(allow_soft_placement=True)
) as sess:
    print(sess.run(c))