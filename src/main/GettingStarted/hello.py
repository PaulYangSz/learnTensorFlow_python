#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is the start of TensorFlow.

How to creat this Pycharm project is more important than the following codes. If you need see the README.
"""

__author__ = 'Paul Yang'

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()

print("sess.run(hello): ", sess.run(hello))

## The Computational Graph
# A computational graph is a series of TensorFlow operations arranged into a graph of nodes.
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)  # also tf.float32 implicitly

print(node1, node2)
print("sess.run([node1, node2]): ", sess.run([node1, node2]))

node3 = tf.add(node1, node2)  # Operations are also nodes.
print(node3)
print("sess.run(node3): ", sess.run(node3))

# A graph can be parameterized to accept external inputs, known as placeholders.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
print("run(adder_node, {a: 3, b: 4.5}): ", sess.run(adder_node, {a: 3, b: 4.5}))
print("run(adder_node, {a: [1, 3], b: [2, 4]}): ", sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# We can make the computational graph more complex by adding another operation.
add_and_triple = adder_node * 3.
print("run(add_and_triple, {a: 3, b: 4.5}): ", sess.run(add_and_triple, {a: 3, b: 4.5}))

# In machine learning we will typically want a model that can take arbitrary inputs
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

'''
Constants are initialized when you call tf.constant, and their value can never change.
By contrast, variables are not initialized when you call tf.Variable.
To initialize all the variables in a TensorFlow program, you must explicitly call a special operation as follows:
'''
init = tf.global_variables_initializer()
sess.run(
    init)  # If don't run this, tensorflow.python.framework.errors_impl.FailedPreconditionError: Attempting to use uninitialized value Variable_1

print("run(linear_model, {x: [1, 2, 3, 4]}): ", sess.run(linear_model, {x: [1, 2, 3, 4]}))

'''
A loss function measures how far apart the current model is from the provided data.
We'll use a standard loss model for linear regression, which sums the squares of the deltas between the current model and the provided data.
linear_model - y creates a vector where each element is the corresponding example's error delta.
We call tf.square to square that error.
Then, we sum all the squared errors to create a single scalar that abstracts the error of all examples using tf.reduce_sum:
'''
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print("run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}): ", sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

'''
 A variable is initialized to the value provided to tf.Variable but can be changed using operations like tf.assign
 We could improve this manually by reassigning the values of W and b to the perfect values of -1 and 1.
'''
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print("run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}): ", sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# tf.train API
'''
TensorFlow can automatically produce derivatives given only a description of the model using the function tf.gradients.
For simplicity, optimizers typically do this for you.
'''
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)  # reset values to incorrect defaults.
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print("run([W, b]): ", sess.run([W, b]))


# 前面这些都是构造具体的Node，然后进行计算，或者对计算结果进行优化，都属于低级别底层的TensorFlow Core操作。
# 下面看一点高级别的TensorFlow库操作：
"""
tf.estimator is a high-level TensorFlow library that simplifies the mechanics of machine learning, including the following:

    running training loops
    running evaluation loops
    managing data sets
tf.estimator defines many common models.
"""
import numpy as np

# Declare list of features. We only have one numeric feature. There are many
# other types of columns that are more complicated and useful.
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# linear classification, and many neural network classifiers and regressors.
# The following code provides an estimator that does linear regression.
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use two data sets: one for training and one for evaluation
# We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# We can invoke 1000 training steps by invoking the  method and passing the
# training data set.
estimator.train(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)


# tf.estimator并不会限制只能使用内置好的模型，如果想自己创建一个没有预先在TensorFlow中创建的模型，依然可以放在tf.estimator中使用。
# 通过低级别的TensorFlow API构造tf.estimator.Estimator的函数入参，然后就能用tf.estimator进行训练预测等。
# Declare list of features, we only have one real-valued feature
def model_fn(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W * features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # EstimatorSpec connects subgraphs we built to the
  # appropriate functionality.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.estimator.Estimator(model_fn=model_fn)
# define our data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1, shuffle=False)

# train
estimator.train(input_fn=input_fn, steps=1000)
# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)
