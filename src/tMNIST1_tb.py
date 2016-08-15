# coding: UTF-8

# tMNIST1_tb.py
# Deep MNIST for Expertsの前半パートのtensorboardを作成するプログラム

import tensorflow as tf

import input_data
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

# 推論 分類結果や回帰結果を予想する部分
def inference(x):

  with tf.name_scope('inference') as scope:

    W = tf.Variable(tf.zeros([784,10]), name="weight")
    b = tf.Variable(tf.zeros([10]), name="bias")
    
    y = tf.nn.softmax(tf.matmul(x,W) + b)

  return y

# 目標値との誤差 最適化したい値
def loss(y, y_):
  with tf.name_scope('loss') as scope:
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    # スカラーサマリーを設定
    tf.scalar_summary("entropy", cross_entropy)
  return cross_entropy

# 最適化アルゴリズム
def training(loss):
  with tf.name_scope('training') as scope:
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
  return train_step

# tensorboardの記録開始
with tf.Graph().as_default():

  x = tf.placeholder("float", shape=[None, 784], name="x")
  y_ = tf.placeholder("float", shape=[None, 10], name="y_")

  # 各パートの計算
  y = inference(x)
  loss = loss(y, y_)
  training_op = training(loss)

  # 全てのサマリーをマージ
  summary_op = tf.merge_all_summaries()

  init = tf.initialize_all_variables()

  # Sessionの起動
  with tf.Session() as sess:

    # SummaryWriterの起動とlogファイルの指定
    summary_writer = tf.train.SummaryWriter('log', graph=sess.graph)

    sess.run(init)

    # 学習
    for i in range(1000):
      batch = mnist.train.next_batch(50)
      feed_dict = {x: batch[0], y_: batch[1]}
      # Sessionの実行
      sess.run(training_op, feed_dict=feed_dict)
      # サマリーの記述
      summary_str = sess.run(summary_op, feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, i)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})