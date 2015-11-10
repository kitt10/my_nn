__author__ = 'kitt'

import numpy as np
import random
from ann_support_tools import sigmoid, sigmoid_prime


class ANNLearning(object):

    def __init__(self, program, learning_name, net):
        self.program = program
        self.name = learning_name
        self.net = net
        self.learning_rate = None
        self.epochs = None
        self.evaluate_epochs = True
        self.epochs_published = None
        self.evaluations = list()

        # Register self to the net
        net.learning = self

    def learn(self, *args):
        raise NotImplementedError('Learning process not defined.')


class BackPropagation(ANNLearning):

    def __init__(self, program, net):
        ANNLearning.__init__(self, program, 'Fast Back-Prop using NumPy', net)
        self.mini_batch_size = None

    def learn(self, training_data):
        self.evaluations = list()
        self.epochs_published = 0

        for epoch in xrange(1, self.epochs+1):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+self.mini_batch_size] for k in xrange(0, len(training_data), self.mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)

            if self.evaluate_epochs:
                self.evaluations.append(self.program.dataset.get_pretty_evaluation_str(epoch))
            else:
                self.evaluations.append('Epoch {0} completed.'.format(epoch))

            self.program.gui.m_w.cb_fake_epoch_done.setChecked(True)

    def update_mini_batch(self, mini_batch):
        nabla_b = [np.zeros(b.shape) for b in self.net.biases]
        nabla_w = [np.zeros(w.shape) for w in self.net.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.net.weights = np.multiply([w-(self.learning_rate/len(mini_batch))*nw for w, nw in zip(self.net.weights, nabla_w)], self.net.synapses_exist)
        self.net.biases = [b-(self.learning_rate/len(mini_batch))*nb for b, nb in zip(self.net.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.net.biases]
        nabla_w = [np.zeros(w.shape) for w in self.net.weights]

        # feedforward
        activation = x
        activations = [x]       # list to store all the activations, layer by layer
        zs = list()             # list to store all the z vectors, layer by layer
        for b, w in zip(self.net.biases, self.net.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = (activations[-1]-y)*sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, len(self.net.n_neurons)):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.net.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return nabla_b, nabla_w
