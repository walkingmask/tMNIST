# coding: UTF-8

# h_conv2_visualize.py
# 畳み込み第二層の畳み込み結果のimage_summaryを作成する
# テスト用に学習回数を少なめに設定(10)

import tensorflow as tf
import input_data
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def variable_summaries(var, name):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

with tf.Graph().as_default():

  with tf.name_scope('input'):
    x = tf.placeholder("float", shape=[None, 784], name="x")
    y_ = tf.placeholder("float", shape=[None, 10], name="y_")

  with tf.name_scope('input_reshape'):
    x_image = tf.reshape(x, [-1,28,28,1])

  with tf.name_scope('1stConvLayer'):
    with tf.name_scope('weights'):
      W_conv1 = weight_variable([5, 5, 1, 32])
      variable_summaries(W_conv1, '1stConvLayer/weights')
    with tf.name_scope('biases'):
      b_conv1 = bias_variable([32])
      variable_summaries(b_conv1, '1stConvLayer/biases')
    with tf.name_scope('Wx_plus_b'):
      preactivate = conv2d(x_image, W_conv1) + b_conv1
      tf.histogram_summary('1stConvLayer/pre_activations', preactivate)
    h_conv1 = tf.nn.relu(preactivate)
    tf.histogram_summary('1stConvLayer/conv', h_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    tf.histogram_summary('1stConvLayer/pool', h_pool1)

  with tf.name_scope('2ndConvLayer'):
    with tf.name_scope('weights'):
      W_conv2 = weight_variable([5, 5, 32, 64])
      variable_summaries(W_conv2, '2ndConvLayer/weights')
    with tf.name_scope('biases'):
      b_conv2 = bias_variable([64])
      variable_summaries(b_conv2, '2ndConvLayer/biases')
    with tf.name_scope('Wx_plus_b'):
      preactivate = conv2d(h_pool1, W_conv2) + b_conv2
      tf.histogram_summary('2ndConvLayer/pre_activations', preactivate)
    h_conv2 = tf.nn.relu(preactivate)
    tf.histogram_summary('2ndConvLayer/conv', h_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    tf.histogram_summary('2ndConvLayer/pool', h_pool2)

  with tf.name_scope('DensLayer'):
    with tf.name_scope('weights'):
      W_fc1 = weight_variable([7 * 7 * 64, 1024])
      variable_summaries(W_fc1, 'DensLayer/weights')
    with tf.name_scope('biases'):
      b_fc1 = bias_variable([1024])
      variable_summaries(b_fc1, 'DensLayer/biases')
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
      tf.histogram_summary('DensLayer/pre_activations', preactivate)
    h_fc1 = tf.nn.relu(preactivate)
    tf.histogram_summary('DensLayer/activations', h_fc1)

  with tf.name_scope('Dropout'):
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  with tf.name_scope('ReadOnlyLayer'):
    with tf.name_scope('weights'):
      W_fc2 = weight_variable([1024, 10])
      variable_summaries(W_fc2, 'ReadOnlyLayer/weights')
    with tf.name_scope('biases'):
      b_fc2 = bias_variable([10])
      variable_summaries(b_fc2, 'ReadOnlyLayer/biases')
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
      tf.histogram_summary('ReadOnlyLayer/pre_activations', preactivate)
    y_conv=tf.nn.softmax(preactivate)
    tf.histogram_summary('ReadOnlyLayer/activations', y_conv)

  with tf.name_scope('cross_entropy'):
    diff = y_ * tf.log(y_conv)
    with tf.name_scope('total'):
      cross_entropy = -tf.reduce_sum(diff)
    tf.scalar_summary('cross entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.scalar_summary('accuracy', accuracy)

  summary_op = tf.merge_all_summaries()
  init = tf.initialize_all_variables()

  with tf.Session() as sess:

    train_writer = tf.train.SummaryWriter('log', graph=sess.graph)
    init.run()

    for i in range(10):
      batch = mnist.train.next_batch(50)
      if i%10 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print "step %d, training accuracy %g"%(i, train_accuracy)

        '''
        visualize conv2
        (?, 14, 14, 64)
        '''
        h_conv2_imgs = tf.slice(h_conv2, [0,0,0,0], [49,-1,-1,-1])
        h_conv2_imgs = tf.split(0, 49, h_conv2_imgs) # (1,14,14,64) * 49
        for i in xrange(49):
          h_conv2_imgs_part = tf.split(3, 64, h_conv2_imgs[i]) # (1,14,14,1) * 64
          for j in xrange(64):
            h_conv2_imgs_part[j] = tf.reshape(h_conv2_imgs_part[j], [14,14,1])
            h_conv2_imgs_part[j] = tf.image.resize_image_with_crop_or_pad(h_conv2_imgs_part[j], 16, 16)
            h_conv2_imgs_part[j] = tf.reshape(h_conv2_imgs_part[j], [1,16,16,1])
          h_conv2_part_img1 = tf.concat(1, h_conv2_imgs_part[0:8])
          h_conv2_part_img2 = tf.concat(1, h_conv2_imgs_part[8:16])
          h_conv2_part_img3 = tf.concat(1, h_conv2_imgs_part[16:24])
          h_conv2_part_img4 = tf.concat(1, h_conv2_imgs_part[24:32])
          h_conv2_part_img5 = tf.concat(1, h_conv2_imgs_part[32:40])
          h_conv2_part_img6 = tf.concat(1, h_conv2_imgs_part[40:48])
          h_conv2_part_img7 = tf.concat(1, h_conv2_imgs_part[48:56])
          h_conv2_part_img8 = tf.concat(1, h_conv2_imgs_part[56:64])
          h_conv2_part_img = tf.concat(2, [h_conv2_part_img1, h_conv2_part_img2, h_conv2_part_img3, h_conv2_part_img4,
                                            h_conv2_part_img5, h_conv2_part_img6, h_conv2_part_img7, h_conv2_part_img8])
          h_conv2_part_img = tf.reshape(h_conv2_part_img, [128,128,1])
          h_conv2_part_img = tf.image.resize_image_with_crop_or_pad(h_conv2_part_img, 132, 132)
          h_conv2_part_img = tf.reshape(h_conv2_part_img, [1,132,132,1])
          h_conv2_imgs[i] = h_conv2_part_img
        h_conv2_img1 = tf.concat(1, h_conv2_imgs[0:7])
        h_conv2_img2 = tf.concat(1, h_conv2_imgs[7:14])
        h_conv2_img3 = tf.concat(1, h_conv2_imgs[14:21])
        h_conv2_img4 = tf.concat(1, h_conv2_imgs[21:28])
        h_conv2_img5 = tf.concat(1, h_conv2_imgs[28:35])
        h_conv2_img6 = tf.concat(1, h_conv2_imgs[35:42])
        h_conv2_img7 = tf.concat(1, h_conv2_imgs[42:49])
        h_conv2_img = tf.concat(2, [h_conv2_img1, h_conv2_img2, h_conv2_img3, h_conv2_img4, h_conv2_img5, h_conv2_img6, h_conv2_img7])

        h_conv2_sum = tf.image_summary("h_conv2_Visualize"+str(i), h_conv2_img, max_images=10)
        h_conv2_sum_str = sess.run(h_conv2_sum, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        train_writer.add_summary(h_conv2_sum_str, i)
      
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

      summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
      train_writer.add_summary(summary_str, i)

    print "test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})