# My Deep MNIST for Experts of TensorFlow Tutorial

ソフトコンピューティングの課題用/自分の勉強のために，TensorFlow Tutorial の "Deep MNIST for Experts" を可能な限りわかりやすく解説したWebページを作成する材料の置き場です．

## ディレクトリ構成

    tMNIST
      |-- README.md
      |-- reference.txt
      |-- img
      |-- logs
      `-- src

### reference.txt  
参考にしたWebページ一覧

### img
入力画像，フィルター，各レイヤーの出力イメージのサンプル．学習前の状態のもの．

### logs
TensorBoard の log．main は20000回の学習後のもので，image_summary なし．その他は各イメージの image_summary ありのサンプルで，学習前の状態のもの．  
(ファイル名にPC名が含まれていたのでリネームしたところtbが動かなくなったため一旦削除)

### src
各種ソースコード．

* tMNIST1_*.py : 前半パートのソースコード．nocom(プレーン)，com(コメント解説付き)，tb(tensorboard用)
* tMNIST2_*.py : 後半パートのソースコード
* *_visualize.py : 各種 image_summary ありの tensorboard の log を作成する

## 実行方法

Python(2.7.11)，virtualenv (15.0.1)，tensorflow (0.8.0) の環境での実行．MNIST_data がディレクトリにない場合は初回実行時にダウンロード作成される．  
*_visualize.py はデフォルトでは学習回数が10回となっているので必要があれば変更する．作成する log の PATH も必要に応じて変更する．

### tMNIST*.pyを実行  

    例
    python tMNIST2_tb.py

### *_visualize.pyを実行

    例
    python python w_conv2_visualize.py

### TensorBoardの起動
    
    例
    tensorboard --log=/path/logs/main

ブラウザで http://0.0.0.0:6006 にアクセス．

## Webページ
(近日公開予定)

## ToDo
* *_visualize のマージ
* gif or 動画の作成
* Webページの作成
    * NNの学習について
    * CNNについて
=======
コメント完成
gif or 動画の作成
Webページの作成
　NNの学習から
>>>>>>> origin/master
