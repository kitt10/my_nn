__author__ = 'kitt'

from ann_support_tools import load_data_wrapper
import numpy as np
from random import uniform


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
        self.n_samples_a_class = None
        self.regeneration_allowed = False
        self.split_ratio = 0.9

        # Net is assaigned later, when created
        self.net = None

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
        pretty_info += '\nSplit ratio: '+str(self.split_ratio)+'\nData re-generation allowed: '+str(self.regeneration_allowed)
        pretty_info += '\nNet input neurons: '+str(self.n_input_neurons)+'\nNet output neurons: '+str(self.n_output_neurons)
        pretty_info += '\nNet recommended structure: '+str(self.common_structure)
        return pretty_info

    def print_self(self):
        print self.get_pretty_info_str()

    def evaluate(self):
        raise NotImplementedError

    def get_pretty_evaluation_str(self, epoch=None):
        result = self.evaluate()
        pretty_eval = '\n---- Evaluation for '+self.name+ '----'
        if epoch:
            pretty_eval += '\nEpoch: '+str(epoch)+'/'+str(self.net.learning.epochs)
            pretty_eval += ', Learning rate: '+str(self.net.learning.learning_rate)
        pretty_eval += '\nCorrect: '+str(result)+'/'+str(self.n_samples_testing)
        pretty_eval += '\nMissed: '+str(self.n_samples_testing-result)+'/'+str(self.n_samples_testing)
        pretty_eval += '\nSuccess rate: '+str((result/float(self.n_samples_testing))*100.0)+' %'
        return pretty_eval


class XOR(ANNDataset):

    def __init__(self):
        ANNDataset.__init__(self, 'XOR')
        self.description = 'A common XOR function.' \
                           '\n[0, 0] -> 1' \
                           '\n[1, 0] -> 0' \
                           '\n[0, 1] -> 0' \
                           '\n[1, 1] -> 1'
        self.n_samples_a_class = 10000
        self.common_structure = '2-2-1'
        self.regeneration_allowed = True
        self.training_data = list()
        self.testing_data = list()
        self.generate_data()

    def generate_data(self):
        train_samples = list()
        train_targets = list()
        test_samples = list()
        test_targets = list()

        for ni in range(self.n_samples_a_class):
            ''' sample for class 0 '''
            x0 = uniform(-0.5, 0.5)
            y0 = uniform(-0.4999, 0.4999)

            ''' sample for class 1 '''
            x1 = uniform(-0.5, 0.5)
            if ni < self.n_samples_a_class/2:
                y1 = uniform(-1.0, -0.5)
            else:
                y1 = uniform(0.5, 1.0)

            ''' train/test split '''
            if ni < self.n_samples_a_class*self.split_ratio:
                train_samples.append([x0, y0])
                train_samples.append([x1, y1])
                train_targets.append([0.0])
                train_targets.append([1.0])
            else:
                test_samples.append([x0, y0])
                test_samples.append([x1, y1])
                test_targets.append([0.0])
                test_targets.append([1.0])

        self.training_data = zip([np.reshape(x, (2, 1)) for x in train_samples], [np.reshape(y, (1, 1)) for y in train_targets])
        self.testing_data = zip([np.reshape(x, (2, 1)) for x in test_samples], [np.reshape(y, (1, 1)) for y in test_targets])
        self.set_data(training_data=self.training_data, testing_data=self.testing_data)

    def evaluate(self):
        test_results = [(self.net.feed_forward_fast(x), y) for (x, y) in self.testing_data]
        return sum(int(abs(x-y) < 0.1) for (x, y) in test_results)


class HandwrittenDigits(ANNDataset):

    def __init__(self):
        ANNDataset.__init__(self, 'Digits (MNIST)')
        self.description = 'Digits recognition : MNIST dataset, reachable from: http://yann.lecun.com/exdb/mnist/'
        training_data, validation_data, testing_data = load_data_wrapper('./data')  # Wrap data from .pkl.gz format
        self.set_data(training_data=training_data, testing_data=testing_data)
        self.common_structure = '784-15-10'
        self.split_ratio = 0.833
        self.regeneration_allowed = False

    def evaluate(self):
        test_results = [(np.argmax(self.net.feed_forward_fast(x)), y) for (x, y) in self.testing_data]
        return sum(int(abs(x-y) < 0.1) for (x, y) in test_results)