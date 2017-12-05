""" Neural Network.

Use high-level API to build neural network. (Custom-made Estimator)

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

This example is using TensorFlow layers, see 'neural_network_raw' example for
a raw implementation with variables.

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data/", one_hot=False)

import tensorflow as tf

# Parameters
learning_rate = 0.1
num_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256  # 1st layer
n_hidden_2 = 256  # 2nd layer
num_input = 784  # MNIST data input: 28 * 28
num_classes = 10  # digits 0~9


# Define the neural network with high-level API(Custom made Estimator)
def construct_neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs.
    x = x_dict['images']
    # Hidden fully connect with 256 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1)
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer

# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode, params):
    # Build the neural network
    logits = construct_neural_net(features)

    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    # If predict mode, early return here.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={'digit': pred_classes})

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    # TF Estimators requires to return a EstimatorSpec, that specify the different ops for training, evaluating, ...
    estim_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op}
    )
    return estim_spec

# Set model params
model_params = {"learning_rate": learning_rate}

# Build the Estimator
model = tf.estimator.Estimator(model_fn=model_fn, model_dir="simple_nn_high", params=model_params)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images},
    y=mnist.train.labels,
    batch_size=batch_size,
    num_epochs=None,
    shuffle=True)
# Train the model
model.train(input_fn=input_fn, steps=num_steps)

# Evaluate the model with test data
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images},
    y=mnist.test.labels,
    batch_size=batch_size,
    shuffle=False
)
# Use Estimator 'evaluate' method
e = model.evaluate(input_fn=input_fn)

print("Test Accuracy: ", e['accuracy'])

