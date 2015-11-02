__author__ = 'kitt'

from ann_datasets import *
from ann_main import *
from ann_learning import *
from ann_gui import *


class Program(object):

    def __init__(self):
        self.datasets = list()
        self.dataset = None
        self.gui = None
        self.net_structure = None
        self.net = None

    def get_datasets(self):
        self.datasets = get_datasets()
        self.dataset = self.datasets[0]
        self.net_structure = [self.dataset.n_input_neurons, self.dataset.n_output_neurons]

    def run_gui(self):
        self.gui = ANNGui(self)

    def generate_net(self):
        self.net = ArtificialNeuralNetwork('Neural Network', self.net_structure)

if __name__ == '__main__':

    ''' Init new program '''
    program = Program()

    ''' Get Datasets '''
    program.get_datasets()

    ''' Start Application '''
    program.run_gui()

    # Create Network
    #net = ArtificialNeuralNetwork('XOR_test', [7, 5, 3])

    # Net learning
    #ANNLearningFastBP(net)
    #net.learning.learn(training_data, epochs=10, mini_batch_size=1, eta=1, test_data=test_data)
    #net.map_params()
    #net.evaluate(X, y, print_all_samples=True)

    # remove edge and evaluate
    #net.synapsesNN[net.neuronsLP[0][1]][net.neuronsLP[1][0]].remove_self()
    #net.synapsesNN[net.neuronsLP[0][0]][net.neuronsLP[1][1]].remove_self()
    #net.evaluate(X, y, print_all_samples=True)

    # re-train and evaluate again
    #net.learning.learn(training_data, epochs=1000, mini_batch_size=1, eta=1, test_data=test_data)
    #net.map_params()
    #net.evaluate(X, y, print_all_samples=True)