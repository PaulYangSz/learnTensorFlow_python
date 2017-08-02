#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TensorFlow’s high-level machine learning API (tf.contrib.learn) makes it easy to configure, train, and evaluate a variety of machine learning models.
In this tutorial, you’ll use tf.contrib.learn to construct a neural network classifier and train it on the Iris data set to predict flower species based on sepal/petal geometry.

You'll write code to perform the following five steps:

1. Load CSVs containing Iris training/test data into a TensorFlow Dataset
2. Construct a neural network classifier
3. Fit the model using the training data
4. Evaluate the accuracy of the model
5. Classify new samples
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib.request

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = "iris_data//iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_data//iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
  # If the training and test sets aren't stored locally, download them.
  if not os.path.exists('iris_data'):
      os.makedirs('iris_data')

  if not os.path.exists(IRIS_TRAINING):
    raw = urllib.request.urlopen(IRIS_TRAINING_URL).read().decode()
    with open(IRIS_TRAINING, "w") as f:
      f.write(raw)

  if not os.path.exists(IRIS_TEST):
    raw = urllib.request.urlopen(IRIS_TEST_URL).read().decode()
    with open(IRIS_TEST, "w") as f:
      f.write(raw)

  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Specify that all features have real-value data
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[10, 20, 10],
                                              n_classes=3,
                                              model_dir="iris_data/tmp/iris_model")
  # Define the training inputs
  def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)

    return x, y

  # Fit model.
  classifier.fit(input_fn=get_train_inputs, steps=2000)

  # Define the test inputs
  def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)

    return x, y

  # Evaluate accuracy.
  ev = classifier.evaluate(input_fn=get_test_inputs, steps=1)
  accuracy_score = ev["accuracy"]
  loss_score = ev['loss']
  global_step = ev['global_step']

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
  print("\nTest Loss: {0:f}\n".format(loss_score))
  print("\nGlobal Steps: {0:d}\n".format(global_step))

  # Classify two new flower samples.
  def new_samples():
    return np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

  predictions = list(classifier.predict(input_fn=new_samples))

  print(
      "New Samples, Class Predictions:    {}\n"
      .format(predictions))

if __name__ == "__main__":
    main()