__author__ = 'kitt'

from ann_datasets import *
from ann_for_results import ArtificialNeuralNetwork
from ann_support_tools import sigmoid, sigmoid_prime
from random import shuffle
from time import time
import numpy as np


class BackPropagation(object):

    def __init__(self, prg, net):
        self.program = prg
        self.net = net
        self.learning_rate = 0.5
        self.epochs = 1000
        self.epochs_published = None
        self.sr_history = list()

        # Register self to the net
        net.learning = self
        self.mini_batch_size = 10

    def learn(self, training_data, convergence_test=False, allowed_fall=0.05, threshold_trained=0.9):
        self.sr_history = list()

        for epoch in xrange(1, self.epochs+1):
            shuffle(training_data)
            mini_batches = [training_data[k:k+self.mini_batch_size] for k in xrange(0, len(training_data), self.mini_batch_size)]
            t0 = time()
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)
            processing_time = time()-t0

            self.program.dataset.evaluate()
            self.sr_history.append(self.program.dataset.success_rate)
            print self.program.dataset.success_rate,
            #print self.program.dataset.get_pretty_evaluation_str(epoch, time=processing_time)

            if convergence_test:
                if self.sr_history[-1] >= threshold_trained-allowed_fall:
                    return True
                else:
                    try:
                        if abs(self.sr_history[-1]-self.sr_history[-5]) <= 0.01:
                            return False
                    except IndexError:
                        pass

            if self.program.dataset.success_rate >= threshold_trained:
                break
        print '\n'

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


class AI4(object):

    def __init__(self):
        self.net = None
        self.net_structure = None
        self.original_n_synapses = None
        self.dataset = None
        self.learning = None
        self.learning_th = 0.9
        self.learning_rate = 0.5

    def convergence_kept(self, tested_net):
        print '-- evaluation after removing synapses'
        self.dataset.net = tested_net
        self.dataset.evaluate()
        print self.dataset.get_pretty_evaluation_str()

        self.learning.net = tested_net
        ret = self.learning.learn(training_data=self.dataset.training_data, convergence_test=True, allowed_fall=0.05, threshold_trained=self.learning_th)

        self.dataset.net = program.net
        self.learning.net = program.net
        return ret


def cut_synapses_l0(a):
    """ cuts synapses with weights lower than the mean weight change """
    mean_change = np.mean([abs(synapse.get_weight()-synapse.init_weight) for synapse in a.synapsesG])
    c = 0
    for synapse in a.synapsesG[:]:
        synapse.set_weight()
        if abs(synapse.weight-synapse.init_weight) < mean_change:
            synapse.remove_self()
            c += 1
    print 'Removed', c, 'synapses by level 0 cutting.'


def cut_synapses_l1(net):
    """ cuts synapses with weights lower than the 1st quartile (25%) weight change """
    q_change = np.percentile([abs(synapse.get_weight()-synapse.init_weight) for synapse in net.synapsesG], 25)

    c = 0
    for synapse in net.synapsesG[:]:
        synapse.set_weight()
        if abs(synapse.weight-synapse.init_weight) < q_change:
            synapse.remove_self()
            c += 1
    print 'Removed', c, 'synapses by level 1 cutting.'


def cut_synapses_l2(net):
    """ cuts synapses with minimal weights change """
    min_change = min([abs(synapse.get_weight()-synapse.init_weight) for synapse in net.synapsesG])

    c = 0
    for synapse in net.synapsesG[:]:
        synapse.set_weight()
        if abs(synapse.weight-synapse.init_weight) == min_change:
            synapse.remove_self()
            c += 1
    print 'Removed', c, 'synapses by level 2 cutting.'

if __name__ == '__main__':

    ''' Init new program '''
    program = AI4()
    #program.net_structure = [2, 15, 1]
    program.net_structure = [784, 15, 10]
    program.net = ArtificialNeuralNetwork(program=program, name=str(program.net_structure), structure=program.net_structure)
    program.original_n_synapses = len(program.net.synapsesG)

    #program.dataset = XOR()
    program.dataset = HandwrittenDigits()
    program.dataset.net = program.net
    program.learning = BackPropagation(prg=program, net=program.net)
    program.learning.learning_rate = program.learning_rate

    program.learning.learn(training_data=program.dataset.training_data, threshold_trained=program.learning_th)
    print 'Learned. Reached', program.dataset.get_pretty_evaluation_str()

    cutting_functions = [cut_synapses_l0, cut_synapses_l1, cut_synapses_l2]
    cutting_level = 0

    min_structure_found = False
    step = 0

    while not min_structure_found:
        step += 1

        print '\n## STEP', step

        print 'Dead neurons (total / by layer):', sum([neuron.dead for neuron in program.net.neuronsG]), '/',
        for layer in program.net.neuronsLP.values():
            print sum([neuron.dead for neuron in layer]),
        print 'Number of synapses:', len(program.net.synapsesG)

        ''' Make a copy of the net '''
        net_tmp = program.net.copy()

        ''' Try to delete some connections '''
        print 'Looking for connections to delete...'
        try:
            cutting_functions[cutting_level](net_tmp)
        except IndexError:
            min_structure_found = True

        ''' Is it still good now, after cutting synapses? '''
        if program.convergence_kept(net_tmp):
            program.net = net_tmp.copy()
            print '\n\n -- Convergence kept in step', step, ': TMP saved. Continue cutting...'
        else:
            cutting_level += 1
            print '\n\n -- Convergence broken in step', step, ': TMP discarded. Cutting level increased to', cutting_level

    print 'Final structure:',
    for layer in program.net.neuronsLP.values():
        print sum([not neuron.dead for neuron in layer]),

    print 'Total number of synapses: ', len(program.net.synapsesG), ':: reduced from', program.original_n_synapses

    print 'Dead neurons (total / by layer):', sum([neuron.dead for neuron in program.net.neuronsG]), '/',
    for layer in program.net.neuronsLP.values():
        print sum([neuron.dead for neuron in layer]),
    program.net.print_net()

    print '\n\nLearning the final structure...'

    program.learning.learn(training_data=program.dataset.training_data, threshold_trained=0.95)