__author__ = 'kitt'


from random import uniform as randfloat
import numpy as np


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


class ArtificialNeuralNetwork(object):

    def __init__(self, name, structure):
        self.name = name
        self.structure = structure
        self.n_neurons_in = structure[0]
        self.nbs_layers_hidden = range(1, len(structure)-1)
        self.n_neurons_hidden = structure[1:-1]
        self.n_neurons_out = structure[-1]

        self.neurons_in = list()
        self.neurons_hidden = dict()
        self.neurons_out = list()

        self.axons = list()
        self.dendrites = list()

    def create_neurons(self):

        # Input neurons
        for neuron_nb in range(1, self.n_neurons_in+1):
            self.neurons_in.append(ArtificialNeuron(self, '0_'+str(neuron_nb)))

        # Hidden neurons
        for layer_nb in self.nbs_layers_hidden:
            self.neurons_hidden[layer_nb] = list()
            for neuron_nb in range(1, self.n_neurons_hidden[layer_nb-1]+1):
                self.neurons_hidden[layer_nb].append(ArtificialNeuron(self, str(layer_nb)+'_'+str(neuron_nb)))

        # Output neurons
        for neuron_nb in range(1, self.n_neurons_out+1):
            self.neurons_out.append(ArtificialNeuron(self, '-1_'+str(neuron_nb)))

    def connect_net_fully_ff(self):
        # Input layer to the first hidden layer
        for neuron_in in self.neurons_in:
            for neuron_hidden_1 in self.neurons_hidden[1]:
                neuron_in.axon.create_dendrite(neuron_hidden_1)

        # Hidden layers together
        for layer_nb in self.nbs_layers_hidden[:-1]:
            for neuron_hidden_a in self.neurons_hidden[layer_nb]:
                for neuron_hidden_b in self.neurons_hidden[layer_nb+1]:
                    neuron_hidden_a.axon.create_dendrite(neuron_hidden_b)

        # The last hidden layer to output layer
        for neuron_hidden_last in self.neurons_hidden[self.nbs_layers_hidden[-1]]:
            for neuron_out in self.neurons_out:
                neuron_hidden_last.axon.create_dendrite(neuron_out)


class ArtificialNeuron(object):

    def __init__(self, net, id):
        self.net = net
        self.id = id
        self.dendrites_in = list()
        self.axon = ArtificialAxon(self.net, self.id+'_a', self)
        self.bias = float()

        self.x = None
        self.y = None


class ArtificialAxon(object):

    def __init__(self, net, id, neuron):
        self.net = net
        self.id = id
        self.neuron = neuron
        self.dendrites_out = list()
        self.value = float()
        self.out_x = None
        self.out_y = None
        self.line = None
        self.net.axons.append(self)

    def create_dendrite(self, target_neuron):
        dendrite = ArtificialDendrite(self.net, self.id+'_d_'+str(len(self.dendrites_out)), self, target_neuron)
        self.dendrites_out.append(dendrite)
        target_neuron.dendrites_in.append(dendrite)


class ArtificialDendrite(object):

    def __init__(self, net, id, axon, target_neuron):
        self.net = net
        self.id = id
        self.axon = axon
        self.target_neuron = target_neuron
        self.axon_out_ind = len(self.axon.dendrites_out)
        self.target_neuron_ind = len(self.target_neuron.dendrites_in)
        self.global_ind = len(self.net.dendrites)
        self.weight = round(randfloat(0.0, 1.0), 2)
        self.line = None
        self.net.dendrites.append(self)

    def remove_self(self):
        del self.net.dendrites[self.global_ind]
        del self.target_neuron.dendrites_in[self.target_neuron_ind]
        del self.axon.dendrites_out[self.axon_out_ind]
        del self