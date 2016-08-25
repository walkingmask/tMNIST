# coding: UTF-8
'''
tMNIST1_com.py
Deep MNIST for Expertsの後半パートのコメント付きプログラム

keywords
* shape: テンソルの形 テンソルの各ランク(入れ子の深さ)に入る値の数
* reduce_sum: テンソルのスカラ総和を求める
* reduce_mean: テンソルのスカラ平均を求める
* tf.cast: Clangのcastと同じである型の変数を別の型の変数として表現する
'''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


'''
重み初期化用の関数
* 切断正規分布(truncated normal distribution)を使った乱数を使ってshapeサイズのtensorを生成する
* stddev: 分布の標準偏差
* 変数の初期値を乱数とする為に，一旦truncated_normalで生成してVariableに変換して渡している
'''
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


'''
バイアス初期化用の関数
* 全て0.1のshapeサイズのテンソルを生成する
* 変数の初期値を0.1とする為に，一旦constantで生成してVariableに変換して渡している
'''
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

'''
畳込み関数
* 畳込み: 画像に適当なフィルタを適用することで特徴を捉える処理
* x: インプットデータの4次元テンソル [batch, 高さ, 幅, チャンネル数(白黒=1, RGB=3)]
* W: 畳込みに使うフィルタの4次元テンソル [高さ, 幅, 入力チャンネル, 出力チャンネル]
* strides: フィルタをずらす間隔 [1固定, 縦の間隔, 横の間隔, 1固定]
* padding: 畳込みを適用するデータへのパッディング SAMEだとゼロパディング
'''
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

'''
最大プーリング関数
* プーリング: 特徴をさらに際立たせて，次元を削減させる処理
             微小な位置と回転に対して不変性を与える
* x: 入力データ 畳込み層からの出力
* ksize: プーリングサイズ この場合2*2
* strides, paddingはconv2dと同じ
'''
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

'''
テンソルのリシェイプ: 畳込み層にテンソルを渡すためにはバッチサイズ，高さ，幅，チャンネル数に変換してやる必要がある
* x_image: 畳込み層に渡すために入力イメージのテンソルを[None, 28, 28, 1]にリシェイプしたもの
'''
x_image = tf.reshape(x, [-1,28,28,1])

'''
第1畳込み層
* 畳込み: 入力と重みを掛け合わせてバイアスを加えたものを活性化関数に渡して，その出力が第1畳込み層の結果となる
* W_conv1: 5*5のフィルタを1チャンネル(白黒)入力に対して32チャンネル出力(フィルタの種類)として生成
* b_conv1: 畳込みフィルタチャンネル数に対応したバイアスの生成
* ReLU: Rectified Linear Unit 活性化(発火)関数の一つ
* conv2dの出力: パッディングしてるので [None, 28, 28, 32]
* プーリングの出力: [None, 14, 14, 32]
'''
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

'''
* 第2畳込み層: 第1畳込み層からの出力を入力として受け取ってさらなる畳込みをする
* h_conv2のshape: [None, 14, 14, 64]
* h_pool2のshape: [None, 7, 7, 64]
* これは，畳込み2層を通すことで28*28の画像を7*7の画像*64個に変換したことを表す
'''
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

'''
密集結合層: イメージ全体の処理をするために1024個のニューロンを畳込み層と全結合させる
W_fc1: 第2畳込み層とニューロンをつなぐ重み
b_fc1: 各ニューロンのバイアス
h_pool2_flat: 第2畳込み層の出力を 2D=[None, 7*7*64] のテンソルに変換したもの
matmul: 行列の掛け算
h_fc1: 第2畳込み層からの入力とW_fc1との積にバイアスユニットを足してReLUに渡したものの返り値
'''
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

'''
ドロップアウト処理: NNの過学習を軽減するために，幾つかのニューロンを無視しながら学習する
keep_prob: ドロップアウトの確率を決める変数
h_fc1_drop: ドロップアウト後のh_fc1
'''
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

'''
読み出し層: ソフトマックス回帰を使った仕上げレイヤー
'''
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

'''
学習と評価の定義
* AdamOptimizer: 確率的勾配降下法を用いたオンライン学習器
'''
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
