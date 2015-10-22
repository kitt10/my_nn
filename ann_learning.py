__author__ = 'kitt'

from scipy import optimize
import numpy as np
import random
from time import time
from ann_support_tools import sigmoid, sigmoid_prime


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


class ANNLearningFastBP(ANNLearning):

    def __init__(self, net):
        ANNLearning.__init__(self, 'Fast Back-Prop using NumPy', net)

    def learn(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for epoch in xrange(1, epochs+1):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data, tolerance=0.1), n_test)
            else:
                if epoch % 100 == 0:
                    print "Epoch {0} complete".format(epoch)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.net.biases]
        nabla_w = [np.zeros(w.shape) for w in self.net.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.net.weights = np.multiply([w-(eta/len(mini_batch))*nw for w, nw in zip(self.net.weights, nabla_w)],
                                       self.net.synapses_exist)
        #self.net.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.net.weights, nabla_w)]
        self.net.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.net.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.net.biases]
        nabla_w = [np.zeros(w.shape) for w in self.net.weights]
        # feedforward
        activation = x
        activations = [x]   # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.net.biases, self.net.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (activations[-1]-y)*sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, len(self.net.n_neurons)):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.net.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data, tolerance):
        test_results = [(np.max(self.net.feed_forward_fast(x)), y) for (x, y) in test_data]
        return sum(int(abs(x-y) < tolerance) for (x, y) in test_results)
