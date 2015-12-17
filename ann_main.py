__author__ = 'kitt'


import numpy as np
from ann_support_tools import sigmoid


class ArtificialNeuralNetwork(object):

    def __init__(self, program, name, structure):
        self.program = program
        self.name = name
        self.n_neurons = structure                  # [n_neurons_in, n_neurons_hidden1, ..., n_neurons_hiddenH, n_neurons_out]
        self.ol_index = len(self.n_neurons)-1       # output layer index
        self.l_indexes = range(self.ol_index+1)     # layers indexes [0, 1, ..., ol_index]
        self.neuronsG = list()                      # [gind]
        self.neuronsLP = dict()                     # [layer_index][position]
        self.synapsesG = list()                     # [gind]
        self.synapsesNN = dict()                    # [neuron_from][neuron_to]
        self.learning = None                        # Learning algorithm for this network

        # Init net parameters randomly as np.arrays
        self.weights = [np.random.randn(n, m) for m, n in zip(self.n_neurons[:-1], self.n_neurons[1:])]
        self.biases = [np.random.randn(n, 1) for n in self.n_neurons[1:]]

        # Coefficients for synapses (useful when removing them)
        self.synapses_exist = [np.ones((n, m)) for m, n in zip(self.n_neurons[:-1], self.n_neurons[1:])]

        # Create units and connect net
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

    def feed_forward_fast(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def map_params(self):
        # Weights
        for synapse in self.synapsesG:
            synapse.set_weight()

        # Biases
        for neuron in self.neuronsG:
            neuron.set_bias()

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
            if not neuron.dead:
                print '\n\n', neuron.id, neuron.activity
                try:
                    print ', synapses_out:', [syn.id for syn in neuron.synapses_out]
                except AttributeError:
                    pass
                except TypeError:
                    pass
                try:
                    print ', synapses_in:', [syn.id for syn in neuron.synapses_in]
                except AttributeError:
                    pass
                except TypeError:
                    pass
            else:
                print 'neuron', neuron.id, 'dead.'

        for synapse in self.synapsesG:
            print '\n\n', synapse.id, synapse.weight

    def copy(self):
        net_copy = ArtificialNeuralNetwork(program=None, name=self.name+'_copy', structure=self.n_neurons[:])

        net_copy.weights = np.array(self.weights, copy=True)
        net_copy.biases = np.array(self.biases, copy=True)

        for synapse in net_copy.synapsesG[:]:
            synapse.set_weight()
            synapse.init_weight = synapse.weight
            if synapse.weight == 0:
                synapse.remove_self()

        return net_copy


class ArtificialNeuron(object):

    def __init__(self, net, layer_ind, layer_id):
        self.net = net
        self.layer_ind = layer_ind
        self.activity = float()
        self.synapses_in = None
        self.synapses_out = None
        self.z = None                   # Unactivated value of neuron (sometimes also 'a')
        self.d = None                   # Delta : for back-propagation
        self.bias = float()
        self.dead = False

        # Register self
        self.gind = len(self.net.neuronsG)
        self.net.neuronsG.append(self)
        self.layer_pos = len(self.net.neuronsLP[self.layer_ind])
        self.net.neuronsLP[self.layer_ind].append(self)
        self.id = layer_id+'.'+str(layer_ind)+'.'+str(self.layer_pos)
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

    def get_bias(self):
        return self.net.biases[self.layer_ind+1][self.layer_pos][0]

    def set_bias(self):
        self.bias = self.net.biases[self.layer_ind-1][self.layer_pos][0]

    def set_bias_b(self, b):
        self.net.biases[self.layer_ind-1][self.layer_pos][0] = b
        self.set_bias()

    def set_dead(self):
        self.dead = True


class ArtificialInputNeuron(ArtificialNeuron):

    def __init__(self, net, layer_ind):
        ArtificialNeuron.__init__(self, net, layer_ind, 'i')
        self.synapses_out = list()

    def activate(self):
        pass

    def feed(self, x):
        self.activity = x

    def get_bias(self):
        return None

    def set_bias(self):
        pass

    def set_bias_b(self, b):
        self.bias = b


class ArtificialHiddenNeuron(ArtificialNeuron):

    def __init__(self, net, layer_ind):
        ArtificialNeuron.__init__(self, net, layer_ind, 'h')
        self.synapses_in = list()
        self.synapses_out = list()
        self.z = float()
        self.d = float()
        self.bias = self.net.biases[self.layer_ind-1][self.layer_pos][0]


class ArtificialOutputNeuron(ArtificialNeuron):

    def __init__(self, net, layer_ind):
        ArtificialNeuron.__init__(self, net, layer_ind, 'o')
        self.synapses_in = list()
        self.z = float()
        self.d = float()
        self.bias = self.net.biases[self.layer_ind-1][self.layer_pos][0]

    def read(self):
        return self.activity


class ArtificialSynapse(object):

    def __init__(self, net, neuron_from, neuron_to):
        self.net = net
        self.neuron_from = neuron_from
        self.neuron_to = neuron_to
        self.weight = self.net.weights[self.neuron_from.layer_ind][self.neuron_to.layer_pos][self.neuron_from.layer_pos]
        self.init_weight = self.weight

        # Register self
        self.gind = len(self.net.synapsesG)
        self.net.synapsesG.append(self)
        self.net.synapsesNN[neuron_from][neuron_to] = self
        self.neuron_from.synapses_out.append(self)
        self.neuron_to.synapses_in.append(self)
        self.id = neuron_from.id+'->'+neuron_to.id
        self.net.synapses_exist[self.neuron_from.layer_ind][self.neuron_to.layer_pos][self.neuron_from.layer_pos] = 1.0

        # Graphics
        self.g_gray_value = None
        self.g_line = None

    def get_weight(self):
        return self.net.weights[self.neuron_from.layer_ind][self.neuron_to.layer_pos][self.neuron_from.layer_pos]

    def set_weight(self):
        self.weight = self.net.weights[self.neuron_from.layer_ind][self.neuron_to.layer_pos][self.neuron_from.layer_pos]

    def set_weight_w(self, w):
        self.net.weights[self.neuron_from.layer_ind][self.neuron_to.layer_pos][self.neuron_from.layer_pos] = w
        self.set_weight()

    def remove_self(self):
        self.net.synapses_exist[self.neuron_from.layer_ind][self.neuron_to.layer_pos][self.neuron_from.layer_pos] = 0.0
        self.set_weight_w(0.0)
        self.neuron_from.synapses_out.remove(self)
        if not self.neuron_from.synapses_out:
            self.neuron_from.set_dead()
        self.neuron_to.synapses_in.remove(self)
        if not self.neuron_to.synapses_in:
            self.neuron_to.set_dead()
        self.net.synapsesG.remove(self)
        del self.net.synapsesNN[self.neuron_from][self.neuron_to]
        del self