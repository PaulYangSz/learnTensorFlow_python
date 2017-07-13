#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
In cases where more feature engineering is needed,
tf.contrib.learn supports using a custom input function (input_fn) to encapsulate the logic for preprocessing and piping data into your models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import functools
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

"""
def my_input_fn():

    # Preprocess your data here...

    # ...then return 1) a mapping of feature columns to Tensors with
    # the corresponding feature data, and 2) a Tensor containing labels
    return feature_cols, labels

feature_cols
    A dict containing key/value pairs that map feature column names to Tensors (or SparseTensors) containing the corresponding feature data.
labels
    A Tensor containing your label (target) values: the values your model aims to predict.
"""

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"


def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels


def main(unused_argv):
  # Load datasets
  training_set = pd.read_csv("boston_data//boston_train.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)
  test_set = pd.read_csv("boston_data//boston_test.csv", skipinitialspace=True,
                         skiprows=1, names=COLUMNS)

  # Set of 6 examples for which to predict median house values
  prediction_set = pd.read_csv("boston_data//boston_predict.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)

  # Feature cols
  feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]

  # Build 2 layer fully connected DNN with 10, 10 units respectively.
  regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                            hidden_units=[10, 10],
                                            model_dir="boston_data//tmp//boston_model")

  # Fit
  regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)
  # regressor.fit(input_fn=functools.partial(input_fn, data_set=training_set), steps=5000)

  # Score accuracy
  ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
  loss_score = ev["loss"]
  print("Loss: {0:f}".format(loss_score))

  # Print out predictions
  y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
  # .predict() returns an iterator; convert to a list and print predictions
  predictions = list(itertools.islice(y, 6))
  print("Predictions: {}".format(str(predictions)))

if __name__ == "__main__":
  tf.app.run()