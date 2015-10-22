__author__ = 'kitt'

from ann_main import *
from ann_learning import *
from ann_support_tools import load_data_wrapper
from ann_gui import ANNGui

if __name__ == '__main__':

    # Create Network
    net = ArtificialNeuralNetwork('DigitClassification_test', [2, 2, 1])

    # Data-set XOR
    X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    y = [[1], [0], [0], [1]]
    train_samples = [np.reshape(x, (2, 1)) for x in X]
    train_targets = [np.reshape(yi, (1, 1)) for yi in y]

    training_data = zip(train_samples, train_targets)
    test_data = training_data

    #training_data, validation_data, test_data = load_data_wrapper('./data')        # MNIST DATASET OF DIGITS

    # Net learning
    ANNLearningFastBP(net)
    net.learning.learn(training_data, epochs=1000, mini_batch_size=1, eta=1, test_data=test_data)
    net.map_params()
    net.evaluate(X, y, print_all_samples=True)

    # remove edge and evaluate
    net.synapsesNN[net.neuronsLP[0][1]][net.neuronsLP[1][0]].remove_self()
    #net.synapsesNN[net.neuronsLP[0][0]][net.neuronsLP[1][1]].remove_self()
    net.evaluate(X, y, print_all_samples=True)

    # re-train and evaluate again
    net.learning.learn(training_data, epochs=1000, mini_batch_size=1, eta=1, test_data=test_data)
    net.map_params()
    net.evaluate(X, y, print_all_samples=True)

    # Visualization
    gui = ANNGui(net)