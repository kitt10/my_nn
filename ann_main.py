__author__ = 'kitt'


from random import uniform as random_float
import numpy as np
from ann_support_tools import sigmoid


class ArtificialNeuralNetwork(object):

    def __init__(self, name, structure):
        self.name = name
        self.n_neurons = structure                  # [n_neurons_in, n_neurons_hidden1, ..., n_neurons_hiddenH, n_neurons_out]
        self.ol_index = len(self.n_neurons)-1       # output layer index
        self.l_indexes = range(self.ol_index+1)     # layers indexes [0, 1, ..., ol_index]
        self.neuronsG = list()                      # [gind]
        self.neuronsLP = dict()                     # [layer_index][position]
        self.synapsesG = list()                     # [gind]
        self.synapsesNN = dict()                    # [neuron_from][neuron_to]
        self.learning = None                        # Learning algorithm for this network

        self.create_neurons()
        self.connect_net_fully_ff()

    def create_neurons(self):
        # Input neurons
        self.neuronsLP[0] = list()
        for i in range(self.n_neurons[0]):
            ArtificialInputNeuron(self, 0)

        # Hidden neurons
        for layer_ind in self.l_indexes[1:-1]:
            self.neuronsLP[layer_ind] = list()
            for i in range(self.n_neurons[layer_ind]):
                ArtificialHiddenNeuron(self, layer_ind)

        # Output neurons
        self.neuronsLP[self.ol_index] = list()
        for i in range(self.n_neurons[self.ol_index]):
            ArtificialOutputNeuron(self, self.ol_index)

    def connect_net_fully_ff(self):
        for layer_ind in self.l_indexes[:-1]:
            for neuron_from in self.neuronsLP[layer_ind]:
                for neuron_to in self.neuronsLP[layer_ind+1]:
                    ArtificialSynapse(self, neuron_from, neuron_to)

    def feed_forward(self, sample):
        """ Feeds the input layer with a sample and returns the output layer values as a NumPy array """

        # Paste data into the net
        for input_neuron, x_i in zip(self.neuronsLP[0], sample):
            input_neuron.feed(x_i)

        # Activate hidden and output neurons gradually
        for layer_ind in self.l_indexes[1:]:
            for neuron in self.neuronsLP[layer_ind]:
                neuron.activate()

        return np.array([output_neuron.read() for output_neuron in self.neuronsLP[self.ol_index]], dtype=float)

    def evaluate(self, X, y, tolerance=0.1, print_all_samples=False):

        print '\n------- Evaluation of net : '+self.name+' (tolerance: '+str(tolerance)+')'

        total_err = float()
        n_correct = 0
        n_miss = 0
        for sample, target in zip(X, y):
            y_hat = self.feed_forward(sample)
            err = 0.5*sum((target-y_hat)**2)
            if err <= tolerance:
                n_correct += 1
            else:
                n_miss += 1
            total_err += err
            if print_all_samples:
                print 'x:', sample, ', actual:', target, ', predict:', y_hat, 'error:', round(err, 6)

        print '\nAverage_error:', round(total_err/len(X), 6)
        print 'n_correct:', str(n_correct)+'/'+str(len(X))
        print 'n_miss:', str(n_miss)+'/'+str(len(X))
        print 'Success:', str((float(n_correct)/len(X))*100.0)+' %\n--------------------------'

    def print_net(self):
        for neuron in self.neuronsG:
            print '\n\n', neuron.id, neuron.activity
            try:
                print ', synapses_out:', [syn.id for syn in neuron.synapses_out]
            except AttributeError:
                pass
            try:
                print ', synapses_in:', [syn.id for syn in neuron.synapses_in]
            except AttributeError:
                pass

        for synapse in self.synapsesG:
            print '\n\n', synapse.id, synapse.weight


class ArtificialNeuron(object):

    def __init__(self, net, layer_ind, layer_id):
        self.net = net
        self.layer_ind = layer_ind
        self.activity = float()
        self.synapses_in = None
        self.synapses_out = None
        self.z = None                   # Unactivated value of neuron (sometimes also 'a')
        self.d = None                   # Delta : for back-propagation
        self.bias = None

        # Register self
        self.gind = len(self.net.neuronsG)
        self.net.neuronsG.append(self)
        self.layer_pos = len(self.net.neuronsLP[self.layer_ind])
        self.net.neuronsLP[self.layer_ind].append(self)
        self.id = layer_id+str(self.layer_pos)
        self.net.synapsesNN[self] = dict()

        # Graphics
        self.g_body = None
        self.g_axon = None
        self.g_x = None
        self.g_y = None
        self.g_axon_x = None
        self.g_axon_y = None

    def activate(self):
        self.z = sum([synapse.neuron_from.activity*synapse.weight for synapse in self.synapses_in]) + self.bias
        self.activity = sigmoid(self.z)


class ArtificialInputNeuron(ArtificialNeuron):

    def __init__(self, net, layer_ind):
        ArtificialNeuron.__init__(self, net, layer_ind, 'i')
        self.synapses_out = list()

    def activate(self):
        pass

    def feed(self, x):
        self.activity = x


class ArtificialHiddenNeuron(ArtificialNeuron):

    def __init__(self, net, layer_ind):
        ArtificialNeuron.__init__(self, net, layer_ind, 'h'+str(layer_ind))
        self.synapses_in = list()
        self.synapses_out = list()
        self.z = float()
        self.d = float()
        self.bias = float()


class ArtificialOutputNeuron(ArtificialNeuron):

    def __init__(self, net, layer_ind):
        ArtificialNeuron.__init__(self, net, layer_ind, 'o')
        self.synapses_in = list()
        self.z = float()
        self.d = float()
        self.bias = float()

    def read(self):
        return self.activity


class ArtificialSynapse(object):

    def __init__(self, net, neuron_from, neuron_to):
        self.net = net
        self.neuron_from = neuron_from
        self.neuron_to = neuron_to
        self.weight = round(random_float(0.0, 1.0), 2)                  # Randomly set weight

        # Register self
        self.gind = len(self.net.synapsesG)
        self.net.synapsesG.append(self)
        self.net.synapsesNN[neuron_from][neuron_to] = self
        self.neuron_from.synapses_out.append(self)
        self.neuron_to.synapses_in.append(self)
        self.id = neuron_from.id+'->'+neuron_to.id

        # Graphics
        self.g_gray_value = None
        self.g_line = None

    def remove_self(self):
        self.neuron_from.synapses_out.remove(self)
        self.neuron_to.synapses_in.remove(self)
        self.net.synapsesG.remove(self)
        del self.net.synapsesNN[self.neuron_from][self.neuron_to]
        del self