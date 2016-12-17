from __future__ import division
import math
import numpy as np

def sigm(data):
    return 1. / (1 + np.exp(-data))

def tanh_opt(data):
    return 1.7159 * np.tanh((2 / 3) * data)

def bsxfun(ifun,ad,bd):
    if len(ad) > len(bd):
        newbd = np.tile(bd, (len(ad), 1))
    else:
        newbd = np.tile(bd, (1, len(ad[0])))
    if ifun is 'plus':
        return np.array(ad) + np.array(newbd)
    if ifun is 'minus':
        return np.array(ad) - np.array(newbd)
    if ifun is 'times':
        return np.array(ad) * np.array(newbd)
    if ifun is 'rdivide':
        return np.array(ad) / np.array(newbd)

def zscore(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data,axis=0)
    data = bsxfun( 'minus', data, mu)
    data = bsxfun('rdivide', data, sigma)
    return data