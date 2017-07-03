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
