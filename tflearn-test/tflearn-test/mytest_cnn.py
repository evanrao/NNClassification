from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import cPickle
import theano.tensor as T
import tensorflow as tf
import numpy as np

# Data loading and preprocessing
#import tflearn.datasets.mnist as mnist
#X, Y, testXa, testYa = mnist.load_data(one_hot=True)

with open('mnist.pkl', 'rb') as f:
    train_set, valid_set, test_set = cPickle.load(f)
f.close
X = train_set[0].reshape([-1, 28, 28, 1])
Y = np.zeros((len(train_set[1]), 10))
for i in range(0, len(train_set[1])):
    Y[i, train_set[1][i]] = 1

testX = test_set[0].reshape([-1, 28, 28, 1])
testY = test_set[1]
testY = np.zeros((len(test_set[1]), 10))
for i in range(0, len(test_set[1])):
    testY[i, test_set[1][i]] = 1

"""
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])
"""


# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')
# fix ----AttributeError: 'module' object has no attribute 'GraphKeys'---------
col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in col:
    tf.add_to_collection(tf.GraphKeys.VARIABLES, x )
# fix -----end

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=20,
           validation_set=({'input': testX}, {'target': testY}),
snapshot_step=100, show_metric=True, run_id='convnet_mnist')
