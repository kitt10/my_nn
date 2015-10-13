__author__ = 'kitt'

from ann_main import *
from PyQt4.QtGui import *
from PyQt4.uic import loadUi


class View(QGraphicsView):
    def __init__(self, parent):
        QGraphicsView.__init__(self, parent)
        self.setScene(QGraphicsScene(self))
        self.setGeometry(50, 20, 700, 400)
        self.neuron_size = 30
        self.axon_len = 20
        self.hor_space = 150
        self.ver_space = 75
        self.pen_neuron = QPen(QColor('DarkBlue'))
        self.pen_neuron.setWidth(2)
        self.pen_axon = QPen(QColor('DarkGreen'))
        self.pen_axon.setWidth(3)
        self.pen_dendrite = QPen(QColor('White'))

    def draw_net(self, net):

        # Input layer
        for i, neuron in enumerate(net.neurons_in):
            neuron.x = 0
            neuron.y = i*self.ver_space
            self.scene().addEllipse(neuron.x, neuron.y, self.neuron_size, self.neuron_size, pen=self.pen_neuron)


        # Hidden layers
        for h in net.nbs_layers_hidden:
            for i, neuron in enumerate(net.neurons_hidden[h]):
                neuron.x = h*self.hor_space
                neuron.y = i*self.ver_space
                self.scene().addEllipse(neuron.x, neuron.y, self.neuron_size, self.neuron_size, pen=self.pen_neuron)

        # Output layer
        for i, neuron in enumerate(net.neurons_out):
            neuron.x = self.hor_space+len(net.nbs_layers_hidden)*self.hor_space
            neuron.y = i*self.ver_space
            self.scene().addEllipse(neuron.x, neuron.y, self.neuron_size, self.neuron_size, pen=self.pen_neuron)

        # Axons
        for axon in net.axons:
            axon.out_x = axon.neuron.x+self.neuron_size+self.axon_len
            axon.out_y = axon.neuron.y+self.neuron_size/2
            axon.line = self.scene().addLine(axon.out_x-self.axon_len, axon.out_y, axon.out_x, axon.out_y, pen=self.pen_axon)

        # Dendrites
        for dendrite in net.dendrites:
            grey_val = 255-dendrite.weight*255
            self.pen_dendrite.setColor(QColor(grey_val, grey_val, grey_val))
            dendrite.line = self.scene().addLine(dendrite.axon.out_x, dendrite.axon.out_y, dendrite.target_neuron.x, dendrite.target_neuron.y+self.neuron_size/2, pen=self.pen_dendrite)



if __name__ == '__main__':
    net = ArtificialNeuralNetwork('first', [4, 3, 5, 2, 1])
    net.create_neurons()
    net.connect_net_fully_ff()

    # Visualization
    app = QApplication([])
    m_w = loadUi('vis.ui')

    # Center the window
    qr = m_w.frameGeometry()
    cp = QDesktopWidget().availableGeometry().center()
    qr.moveCenter(cp)
    m_w.move(qr.topLeft())

    m_w.view = View(m_w)
    m_w.view.draw_net(net)

    m_w.show()
    exit(app.exec_())