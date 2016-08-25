# coding: UTF-8

'''
test.py
Practice of TensorFlow and TesorBoard

Referenced by
http://qiita.com/sergeant-wizard/items/98ce0993a195475fd7a9
http://qiita.com/sergeant-wizard/items/c98597b8add04b8eea0b

Note
"graph_def=sess.graph_def" -> "graph=sess.graph", line 57

'''

import tensorflow as tf

input = [
  [1., 0., 0.],
  [0., 1., 0.],
  [0., 0., 1.]
]

winning_hands = [
  [0., 1., 0.],
  [0., 0., 1.],
  [1., 0., 0.]
]


def inference(input_placeholder):
  with tf.name_scope('inference') as scope:
    W = tf.Variable(tf.zeros([3, 3]), name="weight")
    b = tf.Variable(tf.zeros([3]), name="bias")

    y = tf.nn.softmax(tf.matmul(input_placeholder, W) + b)
  return y

def loss(output, supervisor_labels_placeholder):
  with tf.name_scope('loss') as scope:
    cross_entropy = -tf.reduce_sum(supervisor_labels_placeholder * tf.log(output))
    tf.scalar_summary("entropy", cross_entropy)
  return cross_entropy

def training(loss):
  with tf.name_scope('training') as scope:
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
  return train_step

# MAIN
with tf.Graph().as_default():
  supervisor_labels_placeholder = tf.placeholder("float", [None, 3], name="supervisor_labels_placeholder")
  input_placeholder = tf.placeholder("float", [None, 3], name="input_labels_placeholder")

  feed_dict={input_placeholder: input, supervisor_labels_placeholder: winning_hands}

  output = inference(input_placeholder)
  loss = loss(output, supervisor_labels_placeholder)
  training_op = training(loss)

  summary_op = tf.merge_all_summaries()

  init = tf.initialize_all_variables()

  with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter('data', graph=sess.graph)
    sess.run(init)

    for step in range(1000):
      sess.run(training_op, feed_dict=feed_dict)
      if step % 100 == 0:
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
