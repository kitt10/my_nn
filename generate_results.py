__author__ = 'kitt'

from ann_datasets import *
from ann_for_results import ArtificialNeuralNetwork
from ann_support_tools import sigmoid, sigmoid_prime
from random import shuffle
from time import time
import numpy as np
from ann_plotting import plot_results_for_report_xor
import pickle


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

    def learn(self, training_data, convergence_test=False, allowed_drop=0.05, threshold_trained=0.9):
        self.sr_history = list()

        total_time = 0.0
        for epoch in xrange(1, self.epochs+1):
            shuffle(training_data)
            mini_batches = [training_data[k:k+self.mini_batch_size] for k in xrange(0, len(training_data), self.mini_batch_size)]
            t0 = time()
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)
            total_time += time()-t0

            self.program.dataset.evaluate()
            self.sr_history.append(self.program.dataset.success_rate)
            print self.program.dataset.success_rate,
            #print self.program.dataset.get_pretty_evaluation_str(epoch, time=processing_time)

            if convergence_test:
                if self.sr_history[-1] >= threshold_trained-allowed_drop:
                    return True, epoch, float(total_time)/epoch
                else:
                    try:
                        if abs(self.sr_history[-1]-self.sr_history[-self.program.convergence_history]) <= self.program.convergence_fail_th:
                            return False, epoch, float(total_time)/epoch
                    except IndexError:
                        pass

            if self.program.dataset.success_rate >= threshold_trained:
                break
        print '\n'
        return epoch

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
        self.epochs_needed = None
        self.average_epoch_time = None
        self.n_synapses_to_remove = None
        self.learning_th = 0.9
        self.learning_rate = 0.1
        self.convergence_history = 5
        self.convergence_fail_th = 0.005        # if it doesn't change of 0.5% in five epochs...
        self.allowed_drop = 0.01

    def convergence_kept(self, tested_net):
        print '-- evaluation after removing synapses'
        self.dataset.net = tested_net
        self.dataset.evaluate()
        print self.dataset.get_pretty_evaluation_str()

        self.learning.net = tested_net
        ret, epochs_needed, av_epoch_time = self.learning.learn(training_data=self.dataset.training_data, convergence_test=True, allowed_drop=self.allowed_drop, threshold_trained=self.learning_th)

        self.epochs_needed = epochs_needed
        self.average_epoch_time = av_epoch_time
        self.dataset.net = program.net
        self.learning.net = program.net
        return ret

    def record_net(self, layer0_dict, layer1_dict):
        for input_neuron in self.net.neuronsLP[0]:
            layer0_dict[input_neuron.layer_pos] = [synapse.neuron_to.layer_pos for synapse in input_neuron.synapses_out]

        for hidden_neuron in self.net.neuronsLP[1]:
            layer1_dict[hidden_neuron.layer_pos] = [synapse.neuron_to.layer_pos for synapse in hidden_neuron.synapses_out]


def cut_synapses(net, level):
    """ cuts synapses """
    if level > 0:
        th_change = np.percentile([abs(synapse.get_weight()-synapse.init_weight) for synapse in net.synapsesG], level)
    else:
        th_change = min([abs(synapse.get_weight()-synapse.init_weight) for synapse in net.synapsesG])

    c = 0
    for synapse in net.synapsesG[:]:
        synapse.set_weight()
        if abs(synapse.weight-synapse.init_weight) <= th_change:
            synapse.remove_self()
            c += 1
    program.n_synapses_to_remove = c
    print 'Trying to remove', program.n_synapses_to_remove, 'synapses. May I? Percentile:', level

if __name__ == '__main__':

    ''' Saving stats '''
    stats = dict()

    ''' Init new program '''
    program = AI4()
    program.net_structure = [2, 100, 1]
    #program.net_structure = [784, 15, 10]
    program.net = ArtificialNeuralNetwork(program=program, name=str(program.net_structure), structure=program.net_structure)
    program.original_n_synapses = len(program.net.synapsesG)

    program.dataset = XOR()
    #program.dataset = HandwrittenDigits()
    program.dataset.net = program.net
    program.learning = BackPropagation(prg=program, net=program.net)
    program.learning.learning_rate = program.learning_rate

    program.epochs_needed = program.learning.learn(training_data=program.dataset.training_data, threshold_trained=program.learning_th)
    print 'Learned. Reached', program.dataset.get_pretty_evaluation_str()

    cutting_levels = [75, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    cutting_level_ind = 0

    min_structure_found = False
    step = 0

    ''' Stats '''
    stats[step] = dict()
    stats[step]['n_synapses'] = len(program.net.synapsesG)
    stats[step]['n_neurons'] = sum([not neuron.dead for neuron in program.net.neuronsG])
    stats[step]['n_dead_neurons'] = sum([neuron.dead for neuron in program.net.neuronsG])
    stats[step]['n_dead_neurons_by_layer'] = [sum([neuron.dead for neuron in layer]) for layer in program.net.neuronsLP.values()]
    stats[step]['net_structure'] = [_all-_dead for _all, _dead in zip(program.net_structure, stats[step]['n_dead_neurons_by_layer'])]
    stats[step]['cut_percentile'] = cutting_levels[cutting_level_ind]
    stats[step]['required_accuracy'] = program.learning_th
    stats[step]['allowed_drop'] = program.allowed_drop
    stats[step]['accuracy'] = program.dataset.success_rate
    stats[step]['plus_epochs_needed'] = program.epochs_needed
    stats[step]['average_epoch_time'] = program.average_epoch_time

    # Detailed net's stats
    stats[step]['influenced_neurons_by_i0_neuron'] = dict()
    stats[step]['influenced_neurons_by_h1_neuron'] = dict()
    program.record_net(stats[step]['influenced_neurons_by_i0_neuron'], stats[step]['influenced_neurons_by_h1_neuron'])

    while not min_structure_found:

        ''' Console info '''
        print 'Dead neurons (total / by layer):', stats[step]['n_dead_neurons'], stats[step]['n_dead_neurons_by_layer']
        print 'Number of synapses:', stats[step]['n_synapses'], ', Current structure:', stats[step]['net_structure']

        step += 1
        stats[step] = dict()
        stats[step]['influenced_neurons_by_i0_neuron'] = dict()
        stats[step]['influenced_neurons_by_h1_neuron'] = dict()
        print '\n## STEP', step

        ''' Make a copy of the net '''
        net_tmp = program.net.copy()

        ''' Try to delete some connections '''
        print 'Looking for connections to delete...'
        try:
            cut_synapses(net=net_tmp, level=cutting_levels[cutting_level_ind])
        except IndexError:
            min_structure_found = True

        ''' Is it still good now, after cutting synapses? '''
        if program.convergence_kept(net_tmp):
            program.net = net_tmp.copy()
            stats[step]['accuracy'] = program.dataset.success_rate
            stats[step]['plus_epochs_needed'] = program.epochs_needed
            stats[step]['average_epoch_time'] = program.average_epoch_time
            stats[step]['cut_percentile'] = cutting_levels[cutting_level_ind]
            print '\n\n -- Convergence kept in step', step, ': TMP saved. Continue cutting...'
        else:
            stats[step]['accuracy'] = stats[step-1]['accuracy']
            stats[step]['plus_epochs_needed'] = stats[step-1]['plus_epochs_needed']
            stats[step]['average_epoch_time'] = stats[step-1]['average_epoch_time']
            stats[step]['cut_percentile'] = stats[step-1]['cut_percentile']
            if program.n_synapses_to_remove == 1:
                print '-- Convergence broken in step', step, ': TMP discarded.'
                print 'NOT only a single synapse possible to remove -> final structure found'
                min_structure_found = True
            else:
                cutting_level_ind += 1
                print '\n\n -- Convergence broken in step', step, ': TMP discarded. Cutting level index increased to', cutting_level_ind, \
                    'which is',
                try:
                    print 'percentile', cutting_levels[cutting_level_ind]
                except IndexError:
                    print 'cutting only the synapse with minimum change (only 1)'

        stats[step]['n_synapses'] = len(program.net.synapsesG)
        stats[step]['n_neurons'] = sum([not neuron.dead for neuron in program.net.neuronsG])
        stats[step]['n_dead_neurons'] = sum([neuron.dead for neuron in program.net.neuronsG])
        stats[step]['n_dead_neurons_by_layer'] = [sum([neuron.dead for neuron in layer]) for layer in program.net.neuronsLP.values()]
        stats[step]['net_structure'] = [_all-_dead for _all, _dead in zip(program.net_structure, stats[step]['n_dead_neurons_by_layer'])]
        stats[step]['required_accuracy'] = program.learning_th
        stats[step]['allowed_drop'] = program.allowed_drop
        program.record_net(stats[step]['influenced_neurons_by_i0_neuron'], stats[step]['influenced_neurons_by_h1_neuron'])

    print 'Final structure:',
    for layer in program.net.neuronsLP.values():
        print sum([not neuron.dead for neuron in layer]),

    print 'Total number of synapses: ', len(program.net.synapsesG), ':: reduced from', program.original_n_synapses

    print 'Dead neurons (total / by layer):', sum([neuron.dead for neuron in program.net.neuronsG]), '/',
    for layer in program.net.neuronsLP.values():
        print sum([neuron.dead for neuron in layer]),
    program.net.print_net()

    #print '\n\nLearning the final structure...'

    #program.learning.learn(training_data=program.dataset.training_data, threshold_trained=0.97)

    #print [stats[step]['accuracy'] for step in stats.keys()]
    #print [stats[step]['n_synapses'] for step in stats.keys()]
    #last_step = sorted(stats.keys())[-1]
    #ind_input_neuron = stats[last_step]['influenced_neurons_by_i0_neuron'].keys()[0]
    #ind_hidden_neuron = stats[last_step]['influenced_neurons_by_h1_neuron'].keys()[1]
    #print len(stats[last_step]['influenced_neurons_by_i0_neuron'][ind_input_neuron]), stats[last_step]['influenced_neurons_by_i0_neuron'][ind_input_neuron]
    #print len(stats[last_step]['influenced_neurons_by_h1_neuron'][ind_hidden_neuron]), stats[last_step]['influenced_neurons_by_h1_neuron'][ind_hidden_neuron]

    pickle.dump(stats, open('pickle/xor-2-100-1_test2.p', 'wb'))
    plot_results_for_report_xor(data=stats)