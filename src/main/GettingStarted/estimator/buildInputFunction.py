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

import pandas as pd
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"


def get_input_fn(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),  # 写成data_set[FEATURES]也OK
        y=pd.Series(data_set[LABEL].values),
        num_epochs=num_epochs,
        shuffle=shuffle)


def main(unused_argv):
    # Load datasets
    training_set = pd.read_csv("boston_data/boston_train.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("boston_data/boston_test.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)

    # Set of 6 examples for which to predict median house values
    prediction_set = pd.read_csv("boston_data/boston_predict.csv", skipinitialspace=True,
                                 skiprows=1, names=COLUMNS)

    # Feature cols
    feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

    # Build 2 layer fully connected DNN with 10, 10 units respectively.
    regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[10, 10],
                                          model_dir="boston_data/tmp/boston_model")

    # Train
    regressor.train(input_fn=get_input_fn(training_set), steps=500)

    # Evaluate loss over one epoch of test_set.
    ev = regressor.evaluate(
        input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
    print('evaluate result.key={}'.format(ev.keys()))
    print('ev={}'.format(ev))
    loss_score = ev["loss"]
    print("Loss: {0:f}".format(loss_score))

    # Print out predictions over a slice of prediction_set.
    y = regressor.predict(
        input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False))
    # .predict() returns an iterator of dicts; convert to a list and print
    # predictions
    print("Type of regressor.predict's return: {}".format(type(y)))
    predictions = np.array(list(p["predictions"] for p in itertools.islice(y, 6)))
    print("Predictions: {}".format(str(predictions)))
    print("predictions.shape={}".format(predictions.shape))


if __name__ == "__main__":
    tf.app.run()
