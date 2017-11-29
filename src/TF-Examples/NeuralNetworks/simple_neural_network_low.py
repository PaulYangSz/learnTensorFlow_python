""""
Learn how to use lower level API to predict MNIST

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

#import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data/", one_hot=True)  # class Y is one-hot

import tensorflow as tf

# Parameters
learn_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# Network parameters
n_hidden_1 = 256  # 1st hidden layer
n_hidden_2 = 256  # 2nd hidden layer
num_input = 784  # MNIST data input 28 * 28
num_classes = 10  # 0 - 9 digits

# tf Graph input and output
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])

# Store layers weights and bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
bias = {
    'h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model(Without activated function, because this is real matrix multiply)
def construct_neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), bias['h1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), bias['h2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), bias['out'])
    return out_layer

# Construct model
logits = construct_neural_net(X)
prediction = tf.nn.softmax(logits)

# define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate the model
correct_pred = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initial all variables
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run initializer
    sess.run(init)

    for step in range(1, num_input + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run train_op
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate the batch loss and accuracy
            loss, acc = sess.run(fetches=[loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step = {}, Minibatch loss = {:.4f}, Training Accuracy = {:.4f}".format(step, loss, acc))
    print("Optimization Finished!")

    # Calculate accuracy for the MNIST test images
    print("Test Accuracy: {:.4f}".format(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})))

