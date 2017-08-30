#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
When training a model, it’s often valuable to track and evaluate progress in real time.
In this tutorial, you’ll learn how to use TensorFlow’s logging capabilities and
the Monitor API to audit the in-progress training of a neural network classifier for categorizing irises.
This tutorial builds on the code developed in tf.contrib.learn Quickstart
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib.request

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)  # DEBUG, INFO, WARN, ERROR, and FATAL

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

  validation_metrics = {
      "accuracy":
          tf.contrib.learn.MetricSpec(
              metric_fn=tf.contrib.metrics.streaming_accuracy,
              prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
      "precision":
          tf.contrib.learn.MetricSpec(
              metric_fn=tf.contrib.metrics.streaming_precision,
              prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
      "recall":
          tf.contrib.learn.MetricSpec(
              metric_fn=tf.contrib.metrics.streaming_recall,
              prediction_key=tf.contrib.learn.PredictionKey.CLASSES)
  }

  validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
      test_set.data,
      test_set.target,
      every_n_steps=50,
      metrics=validation_metrics,
      early_stopping_metric='loss',
      early_stopping_metric_minimize=True,
      early_stopping_rounds=200)

  # Specify that all features have real-value data
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[10, 20, 10],
                                              n_classes=3,
                                              model_dir="iris_data/tmp/iris_model",
                                              config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))
  # Define the training inputs
  def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    return x, y

  # Fit model.
  classifier.fit(input_fn=get_train_inputs,
                 steps=300,
                 monitors=[validation_monitor])

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