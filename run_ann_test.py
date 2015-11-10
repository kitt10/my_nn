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
        self.learning = None

    def get_datasets(self):
        self.datasets = [XOR(), HandwrittenDigits()]
        self.dataset = self.datasets[0]
        self.net_structure = [self.dataset.n_input_neurons, self.dataset.n_output_neurons]

    def run_gui(self):
        ANNGui(self)

    def generate_net(self):
        self.net = ArtificialNeuralNetwork(program=self, name='Neural Network', structure=self.net_structure)
        self.learning = BackPropagation(program=self, net=self.net)
        for dataset in self.datasets:
            dataset.net = self.net

    def learn(self):
        self.learning.learn(training_data=self.dataset.training_data)
        self.gui.m_w.cb_fake_training_done.setChecked(True)


if __name__ == '__main__':

    ''' Init new program '''
    program = Program()

    ''' Get Datasets '''
    program.get_datasets()

    ''' Start Application '''
    program.run_gui()