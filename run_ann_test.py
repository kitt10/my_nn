__author__ = 'kitt'

from ann_main import *
from ann_learning import *
from ann_gui import ANNGui

if __name__ == '__main__':

    # Create Network
    net = ArtificialNeuralNetwork('XOR_NN', [2, 3, 1])

    # Data-set XOR
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    y = np.array([[1], [0], [0], [1]], dtype=float)

    # Data-set Test2
    #X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    #y = np.array([[1], [0], [0], [1]], dtype=float)

    # Net learning
    ANNLearningBFGS(net)
    net.learning.learn(X, y)
    net.evaluate(X, y, print_all_samples=False)

    # Visualization
    #gui = ANNGui(net)