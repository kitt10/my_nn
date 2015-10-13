__author__ = 'kitt'


from random import uniform as random_float
import numpy as np


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

        self.create_neurons()

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


class ArtificialNeuron(object):

    def __init__(self, net, layer_ind, layer_id):
        self.net = net
        self.layer_ind = layer_ind
        self.activity = float()

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


class ArtificialInputNeuron(ArtificialNeuron):

    def __init__(self, net, layer_ind):
        ArtificialNeuron.__init__(self, net, layer_ind, 'i')
        self.synapses_out = list()


class ArtificialHiddenNeuron(ArtificialNeuron):

    def __init__(self, net, layer_ind):
        ArtificialNeuron.__init__(self, net, layer_ind, 'h'+str(layer_ind))
        self.synapses_in = list()
        self.synapses_out = list()
        self.z = float()
        self.bias = float()

    def activate(self):
        self.z = sum([synapse.neuron_from.activity*synapse.weight for synapse in self.synapses_in])
        self.activity = sigmoid(self.z)


class ArtificialOutputNeuron(ArtificialNeuron):

    def __init__(self, net, layer_ind):
        ArtificialNeuron.__init__(self, net, layer_ind, 'o')
        self.synapses_in = list()
        self.z = float()
        self.bias = float()

    def activate(self):
        self.z = sum([synapse.neuron_from.activity*synapse.weight for synapse in self.synapses_in])
        self.activity = sigmoid(self.z)


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

        # Graphics
        self.g_gray_value = None
        self.g_line = None

    def remove_self(self):
        self.neuron_from.synapses_out.remove(self)
        self.neuron_to.synapses_in.remove(self)
        self.net.synapsesG.remove(self)
        del self.net.synapsesNN[self.neuron_from][self.neuron_to]
        del self


''' ---------------------------------- STATIC FUNCTIONS ---------------------------------- '''


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))