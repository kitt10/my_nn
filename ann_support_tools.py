__author__ = 'kitt'

import numpy as np
import cPickle
import gzip


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def load_data(data_src):
    f = gzip.open(data_src+'/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper(data_src):
    tr_d, va_d, te_d = load_data(data_src)
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y, 10) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return training_data, validation_data, test_data


def vectorized_result(j, output_neurons):
    e = np.zeros((output_neurons, 1))
    e[j] = 1.0
    return e