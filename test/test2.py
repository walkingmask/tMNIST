# coding: UTF-8
# 各テンソルのshape確認用

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.truncated_normal([5, 784], stddev=0.1)
x_image = tf.reshape(x, [-1,28,28,1])

W = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[32]))

conv = tf.nn.relu(tf.nn.conv2d(x_image, W, strides=[1, 1, 1, 1], padding='SAME') + b)
pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[64]))

conv2 = tf.nn.relu(tf.nn.conv2d(pool, W2, strides=[1, 1, 1, 1], padding='SAME') + b2)
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

h_pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

sess.run(tf.initialize_all_variables())
print sess.run(keep_prob)