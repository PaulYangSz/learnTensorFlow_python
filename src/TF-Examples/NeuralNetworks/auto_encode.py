""" Auto Encoder Example.

Build a 2 layers auto-encoder with TensorFlow to compress images to a
lower latent space and then reconstruct them.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data/", one_hot=True)

# Training Parameters
learning_rate = 0.01
num_steps = 30000
batch_size = 256

display_step = 1000
examples_to_show = 10

# Network Parameters
num_hidden_1 = 256  # 1st layer num features
num_hidden_2 = 128  # 2nd layer num features (the latent dim)
num_input = 784  # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures) 无监督学习
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}


# Build the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Build the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct the model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
# loss_op = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
loss_op = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss=loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
with tf.Session() as sess:
    # Run Initializer
    sess.run(init)

    # Training
    for step in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Run optimization op
        _, loss = sess.run([optimizer, loss_op], feed_dict={X: batch_x})
        # Display logs per step
        if step % display_step == 0 or step == 1:
            print("step {} mini-batch loss = {}".format(step, loss))

    # Test
    # Encode and decode images from test set and visualize their reconstruction
    n_batch = 4
    canvas_orig = np.empty((28 * n_batch, 28 * n_batch))
    canvas_recon = np.empty((28 * n_batch, 28 * n_batch))
    for row_i in range(n_batch):
        print("i = {}".format(row_i))
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n_batch)
        print(batch_x.shape)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # Display original images
        for col_j in range(n_batch):
            # Draw the original digits
            canvas_orig[row_i * 28:(row_i + 1) * 28, col_j * 28:(col_j + 1) * 28] = batch_x[col_j].reshape([28, 28])
        # Display reconstructed images
        for col_j in range(n_batch):
            # Draw the reconstructed digits
            canvas_recon[row_i * 28:(row_i + 1) * 28, col_j * 28:(col_j + 1) * 28] = g[col_j].reshape([28, 28])

    print("Original Images")
    # plt.figure(figsize=(n_batch, n_batch))
    fig, axes = plt.subplots(nrows=1, ncols=2)  # 获得subplot集合
    axes[0].imshow(canvas_orig, origin="upper", cmap="gray")
    axes[0].set_title("Original Images")

    print("Reconstructed Images")
    axes[1].imshow(canvas_recon, origin="upper", cmap="gray")
    axes[1].set_title("Reconstructed Images")
    plt.show()





