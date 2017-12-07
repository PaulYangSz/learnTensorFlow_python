"""
Use Tensorboard to visualize the computation Graph and plot the loss.

Graph and Loss visualization using Tensorboard.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(train_dir="/tmp/tensorflow/mnist/input_data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_epoch = 1
logs_path = '/tmp/TF-Examples/tensorboard_basic/'

# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, 784], name='InputData')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10], name='LabelData')

# Set model weights
W = tf.Variable(tf.zeros([784, 10]), name='Weights')
b = tf.Variable(tf.zeros([10]), name='Bias')

# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
with tf.name_scope('Model'):
    # Model
    pred = tf.nn.softmax(tf.matmul(x, W) + b)
with tf.name_scope('Loss'):
    # Define loss to cross entropy (p * log(1/q))
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
with tf.name_scope('SGD'):
    # Optimizer is SGD
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
with tf.name_scope('Train'):
    train_op = optimizer.minimize(cost)
with tf.name_scope('Accuracy'):
    acc = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
    acc = tf.reduce_mean(tf.cast(acc, dtype=tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar(name="loss", tensor=cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar(name="accuracy", tensor=acc)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # Op to write logs to TensorBoard
    summary_writer = tf.summary.FileWriter(logdir=logs_path, graph=tf.get_default_graph())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op, loss op, and summary nodes
            _, loss, summary = sess.run(fetches=[train_op, cost, merged_summary_op],
                                        feed_dict={x: batch_x, y: batch_y})
            # Write logs at every iteration
            summary_writer.add_summary(summary=summary, global_step=epoch*total_batch + i)
            # Compute average loss
            avg_cost += loss / total_batch
        # Display logs per epoch step
        if (epoch + 1) % display_epoch == 0:
            print("Epoch: {:04d}, cost={:.9f}".format(epoch+1, avg_cost))
    print("Optimization finished!")

    # Test Model
    # Calculate the accuracy
    print("Accuracy: {:.4%}".format(acc.eval({x: mnist.test.images, y: mnist.test.labels})))

    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/TF-Examples/tensorboard_basic/ " \
          "\nThen open http://localhost:6006/ into your web browser")



