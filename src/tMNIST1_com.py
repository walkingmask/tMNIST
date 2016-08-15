# coding: UTF-8

# tMNIST1_com.py
# Deep MNIST for Expertsの前半パートのコメント付きプログラム

# MNISTデータのインポート
import input_data
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

# tensroflowのインポートとinteractiveSessionの作成
import tensorflow as tf
sess = tf.InteractiveSession()

# 入力データと教師データ用の領域
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# ウェイトとバイアス用の変数
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 変数の初期化
sess.run(tf.initialize_all_variables())

# 活性化関数
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 誤差関数
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 最急降下法学習モデル
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 学習
for i in range(1000):
  # 次の50個分のデータをMNISTデータから取得
  batch = mnist.train.next_batch(50)

  # feed_dict(データ)をモデルに渡す
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#  教師データと出力を比較して評価
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))