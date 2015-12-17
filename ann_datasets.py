__author__ = 'kitt'

from ann_support_tools import load_data_wrapper, norm
import numpy as np
from random import uniform, choice, shuffle
from math import sin, cos, pi
from glob import glob
from os import path
from PIL import Image


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
        self.show_allowed = False
        self.split_ratio = 0.9

        self.train_samples = list()
        self.train_targets = list()
        self.test_samples = list()
        self.test_targets = list()

        # Results : TODO: might not be in this class
        self.correctly_classified = 0
        self.success_rate = 0.0

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

    def get_pretty_evaluation_str(self, epoch=None, time=None):
        pretty_eval = '\n---- Evaluation for '+self.name+ '----'
        if epoch:
            pretty_eval += '\nEpoch: '+str(epoch)+'/'+str(self.net.learning.epochs)
            pretty_eval += ', Learning rate: '+str(self.net.learning.learning_rate)
        if time:
            pretty_eval += ', Time [s]: '+str(round(time, 4))
        pretty_eval += '\nCorrect: '+str(self.correctly_classified)+'/'+str(self.n_samples_testing)
        pretty_eval += '\nMissed: '+str(self.n_samples_testing-self.correctly_classified)+'/'+str(self.n_samples_testing)
        pretty_eval += '\nSuccess rate: '+str(self.success_rate*100.0)+' %'
        return pretty_eval


class XOR(ANNDataset):

    def __init__(self):
        ANNDataset.__init__(self, 'xor_001')
        self.description = 'A common XOR function.' \
                           '\n[0, 0] -> 1' \
                           '\n[1, 0] -> 0' \
                           '\n[0, 1] -> 0' \
                           '\n[1, 1] -> 1'
        self.n_samples_a_class = 10000
        self.common_structure = '2-2-1'
        self.regeneration_allowed = True
        self.show_allowed = True
        self.generate_data()

    def generate_data(self):
        self.train_samples = list()
        self.train_targets = list()
        self.test_samples = list()
        self.test_targets = list()
        for ni in range(self.n_samples_a_class):
            ''' sample for class 0 '''
            x0 = uniform(-0.5, 0.5)
            y0 = uniform(-0.49, 0.49)

            ''' sample for class 1 '''
            x1 = uniform(-0.5, 0.5)
            y1 = choice([uniform(-1.0, -0.5), uniform(0.5, 1.0)])

            ''' rotate points in space, 45deg '''
            x0_r = x0*cos(pi/4) - y0*sin(pi/4)
            y0_r = y0*cos(pi/4) + x0*sin(pi/4)
            x1_r = x1*cos(pi/4) - y1*sin(pi/4)
            y1_r = y1*cos(pi/4) + x1*sin(pi/4)

            ''' train/test split '''
            if ni < self.n_samples_a_class*self.split_ratio:
                self.train_samples.append([x0_r, y0_r])
                self.train_samples.append([x1_r, y1_r])
                self.train_targets.append([0.0])
                self.train_targets.append([1.0])
            else:
                self.test_samples.append([x0_r, y0_r])
                self.test_samples.append([x1_r, y1_r])
                self.test_targets.append([0.0])
                self.test_targets.append([1.0])

        training_data = zip([np.reshape(x, (2, 1)) for x in self.train_samples], [np.reshape(y, (1, 1)) for y in self.train_targets])
        testing_data = zip([np.reshape(x, (2, 1)) for x in self.test_samples], [np.reshape(y, (1, 1)) for y in self.test_targets])
        self.set_data(training_data=training_data, testing_data=testing_data)

    def evaluate(self):
        test_results = [(self.net.feed_forward_fast(x), y) for (x, y) in self.testing_data]
        self.correctly_classified = sum(int(abs(x-y) < 0.1) for (x, y) in test_results)
        self.success_rate = float(self.correctly_classified)/self.n_samples_testing


class HandwrittenDigits(ANNDataset):

    def __init__(self):
        ANNDataset.__init__(self, 'Digits (MNIST)')
        self.description = 'Digits recognition : MNIST dataset, reachable from: http://yann.lecun.com/exdb/mnist/'
        training_data, validation_data, testing_data = load_data_wrapper('./data')  # Wrap data from .pkl.gz format
        self.set_data(training_data=training_data, testing_data=testing_data)
        self.common_structure = '784-15-10'
        self.split_ratio = 0.833
        self.regeneration_allowed = False
        self.show_allowed = False

    def evaluate(self):
        test_results = [(np.argmax(self.net.feed_forward_fast(x)), y) for (x, y) in self.testing_data]
        self.correctly_classified = sum(int(abs(x-y) < 0.1) for (x, y) in test_results)
        self.success_rate = float(self.correctly_classified)/self.n_samples_testing


class PenDetection(ANNDataset):

    def __init__(self):
        ANNDataset.__init__(self, 'Pen Detection')
        self.description = 'Detection of pens in clothes out of x-ray images'
        self.common_structure = '1428-10-1'
        self.show_allowed = False
        self.samples = {0.0: list(), 1.0: list()}
        self.fv_length = None
        self.load_data()
        self.split_ratio = 0.9
        self.regeneration_allowed = True
        self.generate_data()

    def evaluate(self):
        test_results = [(self.net.feed_forward_fast(x), y) for (x, y) in self.testing_data]
        self.correctly_classified = sum(int(abs(x-y) < 0.45) for (x, y) in test_results)
        self.success_rate = float(self.correctly_classified)/self.n_samples_testing

    def load_data(self):
        """ Load data from files """

        ''' Load images for class 0 (fail), directory: /data_0/ '''
        for i, path_and_filename in enumerate(glob(path.join('data/pen_detection/data_0/', '*.png'))):
            img_pil = Image.open(path_and_filename).convert('L')
            img_pil.thumbnail((img_pil.size[0]/2, img_pil.size[1]/2))
            self.samples[0.0].append(norm(list(img_pil.getdata()), the_max=255))

        ''' Load images for class 1 (pass), directory: /data_1/ '''
        for i, path_and_filename in enumerate(glob(path.join('data/pen_detection/data_1/', '*.png'))):
            img_pil = Image.open(path_and_filename).convert('L')
            img_pil.thumbnail((img_pil.size[0]/2, img_pil.size[1]/2))
            self.samples[1.0].append(norm(list(img_pil.getdata()), the_max=255))

        self.fv_length = len(self.samples[0.0][0])
        self.common_structure = str(self.fv_length)+'-10-2'

    def generate_data(self):
        """ (re)-generation of dataset """

        shuffle(self.samples[0.0])
        shuffle(self.samples[1.0])

        self.train_samples = list()
        self.train_targets = list()
        self.test_samples = list()
        self.test_targets = list()

        ''' train/test split '''
        for target, samples in self.samples.iteritems():
            for i, sample in enumerate(samples):
                if i < len(samples)*self.split_ratio:
                    self.train_samples.append(sample)
                    self.train_targets.append(target)
                else:
                    self.test_samples.append(sample)
                    self.test_targets.append(target)

        ''' data reshaping '''
        training_data = zip([np.reshape(x, (self.fv_length, 1)) for x in self.train_samples], [np.reshape(y, (1, 1)) for y in self.train_targets])
        testing_data = zip([np.reshape(x, (self.fv_length, 1)) for x in self.test_samples], [np.reshape(y, (1, 1)) for y in self.test_targets])
        self.set_data(training_data=training_data, testing_data=testing_data)