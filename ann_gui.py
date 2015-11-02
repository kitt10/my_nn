__author__ = 'kitt'

from PyQt4.QtGui import *
from PyQt4.uic import loadUi


class ANNGui(object):

    def __init__(self, program):
        app = QApplication([])
        self.m_w = loadUi('gui.ui')
        self.program = program

        # Set up the view
        self.m_w.view = View(self.m_w.widget_view, self.m_w, self.program)

        ''' Center the window '''
        qr = self.m_w.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.m_w.move(qr.topLeft())

        ''' Fill in components '''
        # Dataset Tab
        for ds in self.program.datasets:
            self.m_w.cob_datasets.addItem(ds.name)
        self.m_w.l_dataset_description.setText(self.program.dataset.get_pretty_info_str())

        # Line
        self.m_w.l_data_loaded.setText('Dataset: '+self.program.dataset.name+' ('+str(self.program.dataset.n_samples_training)+
        '/'+str(self.program.dataset.n_samples_testing)+')')
        self.nn_structure_changed()

        # Net Tab
        self.m_w.l_nn_input_neurons.setText('Input neurons: '+str(self.program.dataset.n_input_neurons))
        self.m_w.l_nn_output_neurons.setText('Output neurons: '+str(self.program.dataset.n_output_neurons))

        self.m_w.nn_remove_radio_group = QButtonGroup()
        self.m_w.nn_remove_radio_group.addButton(self.m_w.rb_nn_remove_po)
        self.m_w.nn_remove_radio_group.addButton(self.m_w.rb_nn_remove_ew)

        # Learning Tab
        self.m_w.l_learning_training_samples.setText('Training Samples: '+str(self.program.dataset.n_samples_training))

        ''' Connect settings and listeners '''
        # Datasets
        self.m_w.cob_datasets.currentIndexChanged.connect(self.dataset_changed)

        # Neural Net
        self.m_w.le_hidden_neurons.editingFinished.connect(self.nn_structure_changed)
        self.m_w.pb_generate_network.clicked.connect(self.generate_net)
        self.m_w.pb_nn_remove.clicked.connect(self.remove_synapse)

        # Learning
        self.m_w.sb_learning_epochs.valueChanged.connect(self.learning_settings_changed)
        self.m_w.sb_learning_learning_rate.valueChanged.connect(self.learning_settings_changed)
        self.m_w.sb_learning_minibatch_size.valueChanged.connect(self.learning_settings_changed)
        self.m_w.pb_learning_train_net.clicked.connect(self.train_net)

        # View
        self.m_w.sb_neurons_size.valueChanged.connect(self.view_settings_changed)
        self.m_w.sb_axons_length.valueChanged.connect(self.view_settings_changed)
        self.m_w.sl_horizontal_spaces.valueChanged.connect(self.view_settings_changed)
        self.m_w.sl_vertical_spaces.valueChanged.connect(self.view_settings_changed)
        self.m_w.cb_show_neurons_ids.stateChanged.connect(self.view_settings_changed)
        self.m_w.cb_show_axons_activities.stateChanged.connect(self.view_settings_changed)
        self.m_w.cb_show_synapses_ids.stateChanged.connect(self.view_settings_changed)
        self.m_w.cb_show_synapses_weights.stateChanged.connect(self.view_settings_changed)
        self.m_w.pb_redraw_now.clicked.connect(self.m_w.view.redraw_net)
        self.m_w.cb_redraw_everytime.stateChanged.connect(self.redraw_everytime_changed)

        self.m_w.show()
        exit(app.exec_())

    def dataset_changed(self):
        self.program.dataset = self.program.datasets[self.m_w.cob_datasets.currentIndex()]
        self.m_w.l_dataset_description.setText(self.program.dataset.get_pretty_info_str())
        self.m_w.l_data_loaded.setText('Dataset: '+self.program.dataset.name+' ('+str(self.program.dataset.n_samples_training)+
        '/'+str(self.program.dataset.n_samples_testing)+')')

        # change I/O structure
        self.nn_structure_changed()

        # Net Tab
        self.m_w.l_nn_input_neurons.setText('Input neurons: '+str(self.program.dataset.n_input_neurons))
        self.m_w.l_nn_output_neurons.setText('Output neurons: '+str(self.program.dataset.n_output_neurons))
        self.delete_net()

        # Learning Tab
        self.m_w.l_learning_training_samples.setText('Training Samples: '+str(self.program.dataset.n_samples_training))

    def nn_structure_changed(self):
        self.delete_net()
        self.program.net_structure = [self.program.dataset.n_input_neurons]
        try:
            layers = self.m_w.le_hidden_neurons.text().split(' ')
            for layer in layers:
                self.program.net_structure.append(int(layer))
        except ValueError:
            print 'Try harder.'
        self.program.net_structure.append(self.program.dataset.n_output_neurons)
        self.m_w.l_net_structure.setText('Net Structure: '+str(self.program.net_structure))

        # NN Tab set ranges for deleting synapses
        self.m_w.sb_nn_remove_from_l.setMaximum(len(self.program.net_structure)-1)
        self.m_w.sb_nn_remove_to_l.setMaximum(len(self.program.net_structure))
        self.m_w.sb_nn_remove_from_n.setMaximum(max(self.program.net_structure))
        self.m_w.sb_nn_remove_to_n.setMaximum(max(self.program.net_structure))

    def delete_net(self):
        self.m_w.view.scene().clear()
        self.m_w.l_net_structure.setStyleSheet('color: Maroon;')
        self.m_w.l_training.setText('Training has not began yet.')
        self.m_w.l_training.setStyleSheet('color: Maroon;')
        self.m_w.pb_nn_remove.setEnabled(False)
        self.m_w.pb_learning_train_net.setEnabled(False)
        self.program.net = None

    def generate_net(self):
        self.program.generate_net()
        self.m_w.view.redraw_net()
        self.m_w.l_net_structure.setStyleSheet('color: DarkGreen;')
        self.m_w.pb_nn_remove.setEnabled(True)
        self.m_w.l_learning_algorithm.setText(self.program.net.learning.name)
        self.program.net.learning.epochs = self.m_w.sb_learning_epochs.value()
        self.program.net.learning.learning_rate = self.m_w.sb_learning_learning_rate.value()
        self.program.net.learning.mini_batch_size = self.m_w.sb_learning_minibatch_size.value()
        self.m_w.pb_learning_train_net.setEnabled(True)

    def remove_synapse(self):
        if self.m_w.rb_nn_remove_po.isChecked():
            try:
                neuron_from = self.program.net.neuronsLP[self.m_w.sb_nn_remove_from_l.value()][self.m_w.sb_nn_remove_from_n.value()]
                neuron_to = self.program.net.neuronsLP[self.m_w.sb_nn_remove_to_l.value()][self.m_w.sb_nn_remove_to_n.value()]
                self.program.net.synapsesNN[neuron_from][neuron_to].remove_self()
            except (ValueError, KeyError):
                print 'This synapse does not exists.'
        else:
            for synapse in self.program.net.synapsesG[:]:
                if synapse.weight < self.m_w.dsb_nn_remove_w.value():
                    synapse.remove_self()

        self.m_w.view.redraw_net()

    def learning_settings_changed(self):
        self.program.net.learning.epochs = self.m_w.sb_learning_epochs.value()
        self.program.net.learning.learning_rate = self.m_w.sb_learning_learning_rate.value()
        self.program.net.learning.mini_batch_size = self.m_w.sb_learning_minibatch_size.value()

    def train_net(self):
        self.m_w.l_training.setStyleSheet('color: DarkGreen;')
        self.m_w.l_training.setText('Training...')
        self.program.learn()
        self.m_w.l_training.setText('Last training process done.')

    def view_settings_changed(self):
        self.m_w.view.neuron_size = self.m_w.sb_neurons_size.value()
        self.m_w.view.neuron_font_size = self.m_w.view.neuron_size/5
        self.m_w.view.axon_len = self.m_w.sb_axons_length.value()
        self.m_w.view.hor_space = self.m_w.sl_horizontal_spaces.value()
        self.m_w.view.ver_space = self.m_w.sl_vertical_spaces.value()
        self.m_w.view.show_neurons_ids = self.m_w.cb_show_neurons_ids.isChecked()
        self.m_w.view.show_axons_activities = self.m_w.cb_show_axons_activities.isChecked()
        self.m_w.view.show_synapses_weights = self.m_w.cb_show_synapses_ids.isChecked()
        self.m_w.view.show_synapses_weights = self.m_w.cb_show_synapses_weights.isChecked()
        if self.m_w.cb_redraw_everytime.isChecked():
            self.m_w.view.redraw_net()

    def redraw_everytime_changed(self, state):
        self.m_w.pb_redraw_now.setEnabled(not state)


class View(QGraphicsView):
    def __init__(self, parent, m_w, program):
        QGraphicsView.__init__(self, parent)
        self.m_w = m_w
        self.program = program
        self.setScene(QGraphicsScene(self))
        self.setGeometry(0, 0, parent.width(), parent.height())
        self.neuron_size = self.m_w.sb_neurons_size.value()
        self.neuron_font_size = self.neuron_size/5
        self.axon_len = self.m_w.sb_axons_length.value()
        self.hor_space = self.m_w.sl_horizontal_spaces.value()
        self.ver_space = self.m_w.sl_vertical_spaces.value()
        self.show_neurons_ids = self.m_w.cb_show_neurons_ids.isChecked()
        self.show_axons_activities = self.m_w.cb_show_axons_activities.isChecked()
        self.show_synapses_ids = self.m_w.cb_show_synapses_ids.isChecked()
        self.show_synapses_weights = self.m_w.cb_show_synapses_weights.isChecked()
        self.pen_by_layer = dict()
        self.pen_axon = None
        self.pen_synapse = None
        self.set_pens()

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
        for neuron in self.program.net.neuronsG:
            neuron.g_x = neuron.layer_ind*self.hor_space
            neuron.g_y = neuron.layer_pos*self.ver_space
            neuron.g_axon_x = neuron.g_x+self.neuron_size+self.axon_len
            neuron.g_axon_y = neuron.g_y+self.neuron_size/2
            neuron.g_body = self.scene().addEllipse(neuron.g_x, neuron.g_y, self.neuron_size, self.neuron_size,pen=self.pen_by_layer[neuron.id[0]])
            neuron.g_axon = self.scene().addLine(neuron.g_axon_x-self.axon_len, neuron.g_axon_y, neuron.g_axon_x, neuron.g_axon_y, pen=self.pen_axon)
            if self.show_neurons_ids:
                neuron.g_text = self.scene().addText(neuron.id, QFont('Helvetica', self.neuron_font_size))
                neuron.g_text.translate(neuron.g_x+self.neuron_size/2-15, neuron.g_y+self.neuron_size/2-15)
            if self.show_axons_activities:
                neuron.g_axon_text = self.scene().addText(str(round(neuron.activity, 2)), QFont('Helvetica', 9))
                neuron.g_axon_text.translate(neuron.g_axon_x-self.axon_len, neuron.g_axon_y-25)

        for synapse in self.program.net.synapsesG:
            synapse.g_grey_val = max(min(255-synapse.weight*255, 200), 0)
            self.pen_synapse.setColor(QColor(synapse.g_grey_val, synapse.g_grey_val, synapse.g_grey_val))
            synapse.g_x1 = synapse.neuron_from.g_axon_x
            synapse.g_y1 = synapse.neuron_from.g_axon_y
            synapse.g_x2 = synapse.neuron_to.g_x
            synapse.g_y2 = synapse.neuron_to.g_y+self.neuron_size/2
            synapse.g_line = self.scene().addLine(synapse.g_x1, synapse.g_y1, synapse.g_x2, synapse.g_y2, pen=self.pen_synapse)
            if self.show_synapses_weights:
                synapse.g_text = self.scene().addText(str(round(synapse.weight, 2)), QFont('Helvetica', 8))
                synapse.g_text.translate(synapse.g_x1+(synapse.g_x2-synapse.g_x1)/3-15, synapse.g_y1+(synapse.g_y2-synapse.g_y1)/3-25)

    def redraw_net(self):
        self.scene().clear()
        self.draw_net()