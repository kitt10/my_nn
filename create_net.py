__author__ = 'kitt'

from ann_main import *
from PyQt4.QtGui import *
from PyQt4.uic import loadUi


class View(QGraphicsView):
    def __init__(self, parent, net):
        QGraphicsView.__init__(self, parent)
        self.net = net
        self.setScene(QGraphicsScene(self))
        self.setGeometry(50, 20, 700, 400)
        self.neuron_size = 30
        self.axon_len = 20
        self.hor_space = 150
        self.ver_space = 75
        self.pen_by_layer = dict()
        self.pen_axon = None
        self.pen_synapse = None
        self.set_pens()
        self.draw_net()

    def set_pens(self):
        # Neurons
        self.pen_by_layer['i'] = QPen(QColor('DarkBlue'))
        self.pen_by_layer['h'] = QPen(QColor('DarkRed'))
        self.pen_by_layer['o'] = QPen(QColor('DarkGreen'))

        # Axons
        self.pen_axon = QPen(QColor('DarkGreen'))
        self.pen_axon.setWidth(3)

        # Synapses
        self.pen_synapse = QPen(QColor('White'))

    def draw_net(self):

        for neuron in self.net.neuronsG:
            neuron.g_x = neuron.layer_ind*self.hor_space
            neuron.g_y = neuron.layer_pos*self.ver_space
            neuron.g_axon_x = neuron.g_x+self.neuron_size+self.axon_len
            neuron.g_axon_y = neuron.g_y+self.neuron_size/2
            neuron.g_body = self.scene().addEllipse(neuron.g_x, neuron.g_y, self.neuron_size, self.neuron_size,pen=self.pen_by_layer[neuron.id[0]])
            neuron.g_axon = self.scene().addLine(neuron.g_axon_x-self.axon_len, neuron.g_axon_y, neuron.g_axon_x, neuron.g_axon_y, pen=self.pen_axon)

        for synapse in self.net.synapsesG:
            synapse.g_grey_val = 255-synapse.weight*255
            self.pen_synapse.setColor(QColor(synapse.g_grey_val, synapse.g_grey_val, synapse.g_grey_val))
            synapse.g_line = self.scene().addLine(synapse.neuron_from.g_axon_x, synapse.neuron_from.g_axon_y,
                                                  synapse.neuron_to.g_x, synapse.neuron_to.g_y+self.neuron_size/2,
                                                  pen=self.pen_synapse)

if __name__ == '__main__':
    net = ArtificialNeuralNetwork('first', [2, 3, 1])
    net.connect_net_fully_ff()

    # Visualization
    app = QApplication([])
    m_w = loadUi('vis.ui')

    # Center the window
    qr = m_w.frameGeometry()
    cp = QDesktopWidget().availableGeometry().center()
    qr.moveCenter(cp)
    m_w.move(qr.topLeft())

    m_w.view = View(m_w, net)

    m_w.show()
    exit(app.exec_())