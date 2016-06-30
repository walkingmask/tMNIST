# coding: UTF-8

'''
dMNISTexp2.py
* "Deep MNIST for Experts" の後の(検出率の高い)方のパート CNN
  https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/pros/index.html
* test accuracy 0.9917

Caution
* テスト学習させたい場合は学習のステップ数を100とかにすること！

Referenced
* 日本語訳
  http://qiita.com/uramonk/items/fa9cf1f27bb9387a8c3b
* 日本語解説
  http://qiita.com/haminiku/items/36982ae65a770565458d
* MNIST with tensorboard のチュートリアル
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
* TensroflowのCNN関連関数
  http://qiita.com/tadOne/items/b484ce9f973a9f80036e
* CNNを理解する
  http://tkengo.github.io/blog/2016/03/11/understanding-convolutional-neural-networks-for-nlp/
* CNNを理解する2
  http://aidiary.hatenablog.com/entry/20150626/1435329581
  http://aidiary.hatenablog.com/entry/20150714/1436885749
* Rank, Shapeについて
  http://qiita.com/uyuni/items/5659bb23219a00e7621d
* MNIST for ML Beginnersの数学的な意味合い
  http://neuralnet.hatenablog.jp/entry/2016/04/20/005417
* softmax関数
  http://neuralnet.hatenablog.jp/entry/2016/03/29/090712
* ドロップアウトについて
  http://olanleed.hatenablog.com/entry/2013/12/03/010945

Note
* shape: テンソルの形 テンソルの各ランク(入れ子の深さ)に入る値の数
  shape=[1, 2, 3, 4] であれば tensor=([ [ [ [1,2,3,4],[1,2,3,4],[1,2,3,4] ], [ [1,2,3,4],[1,2,3,4],[1,2,3,4] ] ] ])

ToDo
* どうしてReLUとsoftmaxを使い分けるのだろうか？
* cross entropyとは
* reduce_sum: テンソルのスカラ総和を求める
* reduce_mean: テンソルのスカラ平均を求める
* tf.cast?
* もう一度，NNの学習モデルを，シンプルなモデルでいいから完璧に理解する
  * 重み？バイアス？活性化関数？コスト関数？目的関数？最適化関数？バックプロパゲーション？
* first partも同じくらい詳細に記述する
* tensorboardに出力する
* make non tensorbard ver
'''

import time

# tensorflowとMNISTデータのインポート
import tensorflow as tf
import input_data
mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

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
* 畳込み: 画像に適当なフィルタを適用することで特徴を際立たせる処理
* x: インプットデータの4次元テンソル [batch, 高さ, 幅, チャンネル数(白黒=1, RGB=3)]
* W: 畳込みに使うフィルタの4次元テンソル [高さ, 幅, 入力チャンネル, 出力チャンネル]
* strides: フィルタをずらす間隔 [1固定, 縦の間隔, 横の間隔, 1固定]
* padding: 畳込みを適用するデータへのパッディング SAMEだとゼロパディング
'''
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

'''
最大プーリング関数
* プーリング: 特徴をさらに際立たせる処理
* x: 入力データ 畳込み層からの出力
* ksize: プーリングサイズ この場合2*2
* strides, paddingはconv2dと同じ
'''
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

'''
変数のサマリー作成関数: 受け取った変数のサマリーを生成する
'''
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

  '''
  テンソルのリシェイプ: 畳込み層にテンソルを渡すためにはバッチサイズ，高さ，幅，チャンネル数に変換してやる必要がある
  * x_image: 畳込み層に渡すために入力イメージのテンソルを[None, 28, 28, 1]にリシェイプしたもの
  '''
  with tf.name_scope('input_reshape'):
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

  '''
  * 第2畳込み層: 第1畳込み層からの出力を入力として受け取ってさらなる畳込みをする
  * h_conv2のshape: [None, 14, 14, 64]
  * h_pool2のshape: [None, 7, 7, 64]
  * これは，畳込み2層を通すことで28*28の画像を7*7の画像*64個に変換したことを表す
  '''
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

  '''
  密集結合層: イメージ全体の処理をするために1024個のニューロンを考えて畳込み層と全結合させる
  W_fc1: 第2畳込み層とニューロンをつなぐ重み
  b_fc1: 各ニューロンのバイアス
  h_pool2_flat: 第2畳込み層の出力を 2D=[None, 7*7*64] のテンソルに変換したもの
  matmul: 行列の掛け算
  h_fc1: 第2畳込み層からの入力とW_fc1との積にバイアスユニットを足してReLUに渡したものの返り値
  '''
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

  '''
  ドロップアウト処理: NNの過学習を軽減するために，幾つかのニューロンを無視しながら学習する
  keep_prob: ドロップアウトの確率を決める変数
  h_fc1_drop: ドロップアウト後のh_fc1
  '''
  with tf.name_scope('Dropout'):
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  '''
  読み出し層: ソフトマックス回帰を使った仕上げレイヤー
  '''
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

  '''
  学習と評価の定義
  * AdamOptimizer
  '''
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

  # 学習
  with tf.Session() as sess:

    train_writer = tf.train.SummaryWriter('log', graph=sess.graph)
    init.run()

    for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print "step %d, training accuracy %g"%(i, train_accuracy)
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

      summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
      train_writer.add_summary(summary_str, i)

      time.sleep(0.1)

    print "test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})