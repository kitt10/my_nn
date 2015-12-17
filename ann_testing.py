__author__ = 'kitt'

import numpy as np


class ANNTest(object):

    def __init__(self, program, name):
        """ ANNTest class """

        ''' Register self '''
        self.program = program
        self.gui = self.program.gui
        self.program.test = self

        ''' Load common params '''
        self.n_realizations = int(self.gui.m_w.sb_tests_n_realizations.value())
        self.output_file = str(self.gui.m_w.le_tests_t01_output_file.text())

        ''' Variables '''
        self.name = name

    def run_test(self):
        raise NotImplementedError


class Test01(ANNTest):

    def __init__(self, program):
        ANNTest.__init__(self, program, 'TestParams')

        ''' Load params '''
        self.learning_rate_min = float(self.gui.m_w.dsb_tests_t01_rate_min.value())
        self.learning_rate_max = float(self.gui.m_w.dsb_tests_t01_rate_max.value())
        self.learning_rate_step = float(self.gui.m_w.dsb_tests_t01_rate_step.value())
        self.epochs_max = int(self.gui.m_w.sb_tests_t01_epochs_max.value())

        ''' Variables '''
        self.success_rates = dict()
        self.processing_times = dict()

    def run_test(self):
        for learning_rate in np.arange(self.learning_rate_min, self.learning_rate_max+self.learning_rate_step, self.learning_rate_step):
            self.output_file_append(self.program.dataset.name+';'+str(self.program.net_structure)+';'+str(learning_rate)+';')
            self.success_rates[learning_rate] = dict()
            self.processing_times[learning_rate] = dict()

            for epoch in xrange(1, self.epochs_max+1):
                self.success_rates[learning_rate][epoch] = list()
                self.processing_times[learning_rate][epoch] = list()

            for realization in xrange(1, self.n_realizations+1):
                print 'Testing learning rate '+str(learning_rate)+', Realization '+str(realization)
                self.program.generate_net()
                self.program.learning.learning_rate = learning_rate
                self.program.learning.epochs = self.epochs_max
                self.program.learn(a_test=self)

            for epoch in xrange(1, self.epochs_max+1):
                success_rate = np.mean(self.success_rates[learning_rate][epoch])
                print 'Computed mean of rates', self.success_rates[learning_rate][epoch], ' :: ', success_rate
                time = np.mean(self.processing_times[learning_rate][epoch])
                print 'Computed mean of times', self.processing_times[learning_rate][epoch], ' :: ', success_rate
                self.output_file_append(str(round(success_rate, 4))+'|'+str(round(time, 4))+';')
            self.output_file_append('\n')

    def output_file_append(self, text):
        with open(self.output_file, 'a') as output_file:
            output_file.write(text)

    def append_epoch_results(self, time, epoch):
        self.success_rates[self.program.learning.learning_rate][epoch].append(self.program.dataset.success_rate)
        self.processing_times[self.program.learning.learning_rate][epoch].append(time)
