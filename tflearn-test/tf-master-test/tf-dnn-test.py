#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Example of DNNClassifier for Iris plant dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import cross_validation
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np


def main(unused_argv):
  # Load dataset.

  #iris = learn.datasets.load_dataset('iris')
  #x_train, x_test, y_train, y_test = cross_validation.train_test_split(
  #    iris.data, iris.target, test_size=0.2, random_state=42)
  in_data = np.load('t36v6c6430s.npy')
  x_train, x_test, y_train, y_test = cross_validation.train_test_split(
      in_data[0:len(in_data),0:36], in_data[0:len(in_data),36].astype(np.int32) - 1 , test_size=0.2, random_state=6430)

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  feature_columns = learn.infer_real_valued_columns_from_input(x_train)
  classifier = learn.DNNClassifier(
      feature_columns=feature_columns, hidden_units=[500, 500, 500], n_classes=6)

  # Fit and predict.
  classifier.fit(x_train, y_train, steps=200)
  predictions = list(classifier.predict(x_test, as_iterable=True))
  score = metrics.accuracy_score(y_test, predictions)
  print('Accuracy: {0:f}'.format(score))


if __name__ == '__main__':
  tf.app.run()
