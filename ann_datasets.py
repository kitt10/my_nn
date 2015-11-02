__author__ = 'kitt'

from ann_support_tools import load_data_wrapper
import numpy as np


class ANNDataset(object):

    def __init__(self, name):
        self.name = name
        self.description = None
        self.training_data = None
        self.testing_data = None
        self.n_samples_training = None
        self.n_samples_testing = None
        self.n_input_neurons = None
        self.n_output_neurons = None
        self.common_structure = None

    def set_data(self, training_data, testing_data=None):
        self.training_data = training_data
        self.n_samples_training = len(self.training_data)
        if testing_data:
            self.testing_data = testing_data
            self.n_samples_testing = len(self.testing_data)
        else:
            self.n_samples_testing = 0
        self.n_input_neurons = len(self.training_data[0][0])
        self.n_output_neurons = len(self.training_data[0][1])

    def get_pretty_info_str(self):
        pretty_info = '\n---- Dataset '+self.name+' ----'
        pretty_info += '\n'+self.description+'\n'
        pretty_info += '\nTraining samples: '+str(self.n_samples_training)+'\nTesting samples: '+str(self.n_samples_testing)
        pretty_info += '\nNet input neurons: '+str(self.n_input_neurons)+'\nNet output neurons: '+str(self.n_output_neurons)
        pretty_info += '\nNet recommended structure: '+str(self.common_structure)
        return pretty_info

    def print_self(self):
        print self.get_pretty_info_str()


def get_datasets():
    datasets = list()

    # XOR
    ds = ANNDataset('XOR')
    ds.description = 'A common XOR function.\n[0, 0] -> 1\n[1, 0]-> 0\n[0, 1] -> 0\n[1, 1] -> 1'

    samples = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    targets = [[1], [0], [0], [1]]
    data = zip([np.reshape(x, (2, 1)) for x in samples], [np.reshape(y, (1, 1)) for y in targets])
    ds.set_data(training_data=data, testing_data=data)
    ds.common_structure = '2-2-1'
    datasets.append(ds)

    # Digit recognition (MNIST)
    ds = ANNDataset('Digits (MNIST)')
    ds.description = 'Digits recognition : MNIST dataset, reachable here: http://yann.lecun.com/exdb/mnist/'
    training_data, validation_data, test_data = load_data_wrapper('./data')        # Wrap data from .pkl.gz format
    ds.set_data(training_data=training_data, testing_data=test_data)
    ds.common_structure = '784-15-10'
    datasets.append(ds)

    return datasets