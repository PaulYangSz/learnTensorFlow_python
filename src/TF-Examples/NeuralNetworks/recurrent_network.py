""" Recurrent Neural Network.

A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data/", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 56  # MNIST data input (img shape: 28*28)
timesteps = 14  # timesteps
num_hidden = 128  # hidden layer num of features
num_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    # 将输入的tensor x进行rank - 1操作，
    #  For example, given a tensor of shape `(A, B, C, D)`;
    #  If `axis == 1` then the i'th tensor in `output` is the slice
    #    `value[:, i, :, :]` and each tensor in `output` will have shape `(A, C, D)`.
    # x的类型是list，长度是timesteps, list中的每个元素大小为[batch_size, num_input]
    x = tf.unstack(x, timesteps, axis=1)
    print(tf.Print(x, [x]))

    # Define a LSTM cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_units=num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(cell=lstm_cell, inputs=x, dtype=tf.float32)
    print(tf.Print(outputs, [outputs]))

    # Linear activation, using rnn inner loop last output
    return tf.matmul(a=outputs[-1], b=weights['out']) + biases['out']


logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define the loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(input=prediction, axis=1), tf.argmax(Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize all variables
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run init
    sess.run(init)

    for step in range(1, training_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run train_op
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            print(tf.Print(batch_x, [batch_x]))
            # Calculate the batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step {}, Minibatch Loss = {:.4f}, Training Accuracy = {:.3f}".format(step, loss, acc))
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
