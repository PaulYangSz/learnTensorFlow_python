#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is the start of TensorFlow.

How to creat this Pycharm project is more important than the following codes. If you need see the README.
'''

__author__ = 'Paul Yang'


import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()

print(sess.run(hello))
