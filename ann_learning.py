__author__ = 'kitt'

from scipy import optimize
import numpy as np
from time import time
from ann_support_tools import sigmoid_prime


class ANNLearning(object):

    def __init__(self, learning_name, net):
        self.name = learning_name
        self.net = net
        self.learning_rate = float()

        # Register self to the net
        net.learning = self

    def learn(self, *args):
        raise NotImplementedError('Learning process not defined.')

    def compute_deltas(self, errs):
        # Output layer
        for output_neuron, err in zip(self.net.neuronsLP[self.net.ol_index], errs):
            output_neuron.d = sigmoid_prime(output_neuron.z)*err

        # Hidden layers
        for layer_ind in self.net.l_indexes[1:-1]:
            for neuron in self.net.neuronsLP[layer_ind]:
                neuron.d = sigmoid_prime(neuron.z)*sum([synapse.weight*synapse.neuron_to.d for synapse in neuron.synapses_out])

    def update_weights(self):
        for synapse in self.net.synapsesG:
            synapse.weight += self.learning_rate*synapse.neuron_from.activity*synapse.neuron_to.d


class ANNLearningBasic(ANNLearning):

    def __init__(self, net):
        ANNLearning.__init__(self, 'Basic Back-Prop (Koh)', net)
        self.epochs = int()
        self.evaluate_each_epoch = False
        self.report = True
        self.current_error = float()

    def learn(self, samples, targets, learning_rate=0.1, epochs=10000, evaluate_each_epoch=False, report=True):
        print ':: Learning called '+self.name+' for net called '+self.net.name+' has just started.'
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.evaluate_each_epoch = evaluate_each_epoch
        self.report = report
        self.back_prop(samples, targets)

    def back_prop(self, X, y):
        """ Basic back-propagation algorithm (Koh), X and y are np.arrays """

        t0 = time()
        error_before = float()
        for i in range(self.epochs):
            self.current_error = 0
            for sample, target in zip(X, y):
                y_hat = self.net.feed_forward(sample)
                errs = target-y_hat
                self.compute_deltas(errs)
                self.update_weights()
                self.current_error += 0.5*sum(errs**2)

            if self.evaluate_each_epoch:
                print 'Epoch '+str(i+1)+' :: error: '+str(self.current_error)
            if i == 0:
                error_before = self.current_error

        learning_time = time()-t0
        if self.report:
            self.report_result(error_before, learning_time)

    def report_result(self, error_before, learning_time):
        print '\n----------------------------------------------------------------------------------'
        print 'Learning results :: error before:', round(error_before, 4), ', error after:', \
            round(self.current_error, 4), ', time [s]:', round(learning_time, 4)
        print '----------------------------------------------------------------------------------\n'


class ANNLearningBFGS(ANNLearning):

    def __init__(self, net):
        ANNLearning.__init__(self, 'Efficient Back-Prop : BFGS', net)

    def learn(self, samples, targets, learning_rate=1):
        self.learning_rate = learning_rate
        params0 = self.get_params()
        options = {'maxiter': 200, 'disp': False}

        optimization_res = optimize.minimize(self.cost_fun_wrapper, params0, jac=True, method='BFGS',
                                             args=(samples, targets), options=options, callback=self.set_params)
        self.set_params(optimization_res.x)
        print optimization_res

    def get_params(self):
        return np.array([synapse.weight for synapse in self.net.synapsesG], dtype=float)

    def set_params(self, params):
        for neuron, param in zip(self.net.synapsesG, params):
            neuron.weight = param

    def cost_function(self, X, y):
        err = float()
        for sample, target in zip(X, y):
            y_hat = self.net.feed_forward(sample)
            err += (target-y_hat)**2
        return 0.5*err

    def compute_gradients(self, X, y):
        for sample, target in zip(X, y):
            y_hat = self.net.feed_forward(sample)
            self.compute_deltas(target-y_hat)
            self.update_weights()
        return np.array([synapse.neuron_from.activity*synapse.neuron_to.d for synapse in self.net.synapsesG], dtype=float)

    def cost_fun_wrapper(self, params, X, y):
        self.set_params(params)
        cost = self.cost_function(X, y)
        grad = self.compute_gradients(X, y)
        return cost, grad