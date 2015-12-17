__author__ = 'kitt'

from ann_datasets import *
from ann_main import *
from ann_learning import *
from ann_testing import *
from ann_gui import *
from threading import Thread, ThreadError


class Program(object):

    def __init__(self):
        self.datasets = list()
        self.dataset = None
        self.gui = None
        self.net_structure = None
        self.net = None
        self.learning = None
        self.test = None

    def get_datasets(self):
        self.datasets = [XOR(), HandwrittenDigits(), PenDetection()]
        self.dataset = self.datasets[0]
        self.net_structure = [self.dataset.n_input_neurons, self.dataset.n_output_neurons]

    def run_gui(self):
        ANNGui(self)

    def generate_net(self):
        self.net = ArtificialNeuralNetwork(program=self, name='Neural Network', structure=self.net_structure)
        self.learning = BackPropagation(program=self, net=self.net)
        for dataset in self.datasets:
            dataset.net = self.net

    def learn(self, a_test=None):
        self.learning.learn(training_data=self.dataset.training_data, a_test=a_test)
        self.gui.m_w.cb_fake_training_done.setChecked(True)

    def run_test(self, test_nb):
        if test_nb == 1:
            Test01(program=self)

        try:
            t = Thread(target=self.test.run_test, args=())
            t.setDaemon(True)
            t.start()
        except ThreadError:
            print 'Error in a new thread for Test', test_nb


if __name__ == '__main__':

    ''' Init new program '''
    program = Program()

    ''' Get Datasets '''
    program.get_datasets()

    ''' Start Application '''
    program.run_gui()