"""
Graph and Loss visualization using Tensorboard.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Going deeper into Tensorboard; visualize the variables, gradients, and more...

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
display_step = 1
logs_path = '/tmp/TF-Examples/tensorboard_advanced/'

# Network Parameters
n_hidden_1 = 256  # 1st layer number of features
n_hidden_2 = 256  # 2nd layer number of features
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, 784], name='InputData_x')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10], name='LabelData_y')

# Store layers weight & bias
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='W1'),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='W2'),
    'w3': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='W3')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
    'b3': tf.Variable(tf.random_normal([n_classes]), name='b3')
}


# Create Model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer 1 with Relu activation
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Create a summary to visualize the fist layer Relu activation
    tf.summary.histogram(name="relu_1", values=layer_1)
    # Hidden layer 2 with Relu activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Create another summary to visualize the second layer Relu activation
    tf.summary.histogram(name="relu_2", values=layer_2)
    # Output layer
    out_layer = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
    return out_layer

# Encapsulating all ops into scopes, making TensorBoard's graph more conveniently to visualize
with tf.name_scope(name='Model'):
    # Build model
    pred = multilayer_perceptron(x, weights, biases)

with tf.name_scope(name='Loss'):
    # Softmax Cross Entropy (cost function)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

with tf.name_scope(name='SGD'):
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Op to calculate every variable gradient
    grads = tf.gradients(ys=loss, xs=tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    # Op to update all variables according to their gradient
    apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

with tf.name_scope(name='Accuracy'):
    # Accuracy
    acc = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
    acc = tf.reduce_mean(tf.cast(acc, dtype=tf.float32))

# Initialize the variables
init = tf.global_variables_initializer()

# Create a summary to monitor cost scalar
tf.summary.scalar(name='loss', tensor=loss)
# Create a summary to monitor accuracy scalar
tf.summary.scalar(name='accuracy', tensor=acc)

# Create summary to visualize weights
for var in tf.trainable_variables():
    tf.summary.histogram(name=var.name, values=var)
# Summary all gradients
for grad, var in grads:
    tf.summary.histogram(var.name + '/gradient', grad)

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Start Training
with tf.Session() as sess:
    # Run initializer
    sess.run(init)

    # op to write logs to TensorBoard
    summary_writer = tf.summary.FileWriter(logdir=logs_path, graph=tf.get_default_graph())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # loop overall batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (BP), loss op, summary nodes
            _, c, summary = sess.run([apply_grads, loss, merged_summary_op], feed_dict={x: batch_xs, y: batch_ys})
            # Write log at every iteration
            summary_writer.add_summary(summary=summary, global_step=epoch * total_batch + i)
            # Compute average cost
            avg_cost += c / total_batch
        # Display logs per epoch
        if (epoch + 1) % display_step == 0:
            print("Epoch: {:04d}, loss = {:.9f}".format(epoch+1, avg_cost))

    print("Optimization Finished!")

    # Test Model
    # Calculate the accuracy
    print("Accuracy: {:.4%}".format(acc.eval({x: mnist.test.images, y: mnist.test.labels})))

    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/TF-Examples/tensorboard_advanced " \
          "\nThen open http://localhost:6006/ into your web browser")





