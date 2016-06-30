# coding: UTF-8

'''
dMNISTexp1.py
* "Deep MNIST for Experts" の最初(検出率の低い方)のパート
  https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/pros/index.html

Referenced
* http://qiita.com/uramonk/items/fa9cf1f27bb9387a8c3b
* http://qiita.com/haminiku/items/36982ae65a770565458d

Note
* Tensorboard用のSummaryを出力
* 以下に基づきコードを分割
  http://qiita.com/sergeant-wizard/items/98ce0993a195475fd7a9

'''

# tensroflowのインポート
import tensorflow as tf

# MNIST入力データのインポート
import input_data
mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

# 推論 分類結果や回帰結果を予想する部分
def inference(x):

  # tensorboard用にネームスコープを追加
  with tf.name_scope('inference') as scope:

    # ウェイトとバイアス用の変数の用意
    # tensorboardのサマリーようにnameを設定
    W = tf.Variable(tf.zeros([784,10]), name="weight")
    b = tf.Variable(tf.zeros([10]), name="bias")
    
    # 予測のための回帰モデル(softmax regression)の実装
    # softmaxはsigmoidの多変量版
    y = tf.nn.softmax(tf.matmul(x,W) + b)

  return y

# 目標値との誤差 最適化したい値
def loss(y, y_):
  with tf.name_scope('loss') as scope:

    # 評価関数の実装
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    # tensorboard用にスカラーサマリーを設定
    tf.scalar_summary("entropy", cross_entropy)

  return cross_entropy

# 最適化アルゴリズム
def training(loss):
  with tf.name_scope('training') as scope:

    # ステップ長0.01の最急降下法を使って最適化する学習モデルの実装
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

  return train_step

# MAIN
# グラフの起動
with tf.Graph().as_default():

  # 入力データと教師データ用の領域の用意
  x = tf.placeholder("float", shape=[None, 784], name="x")
  y_ = tf.placeholder("float", shape=[None, 10], name="y_")

  # 各パートの計算
  y = inference(x)
  loss = loss(y, y_)
  training_op = training(loss)

  # 全てのサマリーをマージ
  summary_op = tf.merge_all_summaries()

  # 変数の初期化の実装
  init = tf.initialize_all_variables()

  # Sessionの起動
  with tf.Session() as sess:

    # SummaryWriterの起動とlogファイルの指定
    summary_writer = tf.train.SummaryWriter('log', graph=sess.graph)

    # 変数の初期化
    sess.run(init)

    # 学習
    for i in range(1000):
      # 次の50個分のデータをMNISTデータから取得
      batch = mnist.train.next_batch(50)
      # feed_dict(sessに渡す値)を定義
      feed_dict = {x: batch[0], y_: batch[1]}
      # Sessionの実行
      sess.run(training_op, feed_dict=feed_dict)

      # サマリーの記述
      summary_str = sess.run(summary_op, feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, i)

    # 教師データと出力を比較して評価
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # 評価値の出力
    print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})