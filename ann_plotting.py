__author__ = 'kitt'

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import pickle


def read_file(input_file):
    data = list()
    with open(input_file, 'r') as the_file:
        lines = the_file.read().split('\n')
    for line in lines:
        data.append(line.split(';'))
    return data


def plot_t01_sr_lr_ep(input_file):
    try:
        data = read_file(input_file=input_file)
        data_for_plot = list()
        for line in data[:-1]:
            data_for_plot.append([float(d.split('|')[0]) for d in line[3:-1]])

        xticks_scale = 100
        yticks_scale = 1

        # Imshow success of classification
        plt.imshow(data_for_plot, interpolation='nearest', aspect='auto', vmin=0.0, vmax=1.0, origin='lower')
        plt.colorbar()
        plt.xlabel('Epochs')
        plt.ylabel('Learning rate')
        plt.suptitle('Classification Success Rate')
        xticks = [ep for ep in range(1, len(data_for_plot[0])+1) if ep % xticks_scale == 0]
        xticks.insert(0, 1)
        xticks_positions = xticks
        plt.xticks(xticks_positions, xticks)
        yticks = [line[2] for i, line in enumerate(data[:-1]) if i % yticks_scale == 0]
        yticks_positions = [i*yticks_scale for i in range(len(yticks))]
        plt.yticks(yticks_positions, yticks)
        plt.xlim([-xticks_scale, xticks_positions[-1]+xticks_scale])
        plt.ylim([-yticks_scale, yticks_positions[-1]+yticks_scale])

        # Annotation
        for x in xticks_positions:
            for y in yticks_positions:
                plt.annotate(str(round(data_for_plot[y][x-1], 2)), xy=(x, y), horizontalalignment='center', verticalalignment='center', fontsize=10)

        plt.grid()
        plt.show()

        # Imshow processing time
        plt.figure()
        data_for_plot = list()
        for line in data[:-1]:
            data_for_plot.append([float(d.split('|')[1]) for d in line[3:-1]])
        plt.imshow(data_for_plot, interpolation='nearest', aspect='auto', origin='lower')
        plt.colorbar()
        plt.xlabel('Epochs')
        plt.ylabel('Learning rate')
        plt.suptitle('Learning Processing Time')
        yticks = [line[2] for i, line in enumerate(data[:-1]) if i % yticks_scale == 0]
        plt.yticks([i*yticks_scale for i in range(len(yticks))], yticks)
        plt.grid()
        plt.show()
    except IOError:
        print 'Tento soubor neexistuje.'


def plot_results_for_report(data):
    """ plotting for XOR problem """

    cutting_steps = sorted(data.keys())

    """
    ''' What do particular network parts mean '''
    neurons_ids_input_layer = data[cutting_steps[-1]]['influenced_neurons_by_i0_neuron'].keys()
    neurons_ids_hidden_layer = data[cutting_steps[-1]]['influenced_neurons_by_h1_neuron'].keys()
    hiddens_followed_by_digit = dict()

    for digit in range(10):
        hiddens_followed_by_digit[digit] = list()
        for hidden in neurons_ids_hidden_layer:
            if digit in data[cutting_steps[-1]]['influenced_neurons_by_h1_neuron'][hidden]:
                hiddens_followed_by_digit[digit].append(hidden)

        digit_interest_image = [255 if set(hiddens_followed_by_digit[digit]).intersection(data[cutting_steps[-1]]['influenced_neurons_by_i0_neuron'][key]) else 0 for key in neurons_ids_input_layer]
        img = Image.new("L", (28, 28), "white")
        img.putdata(digit_interest_image)
        img.save('results_plots/mnist_pixels_interesting_for_digit_'+str(digit)+'.png')

    ''' n_synapses and accuracy vs. cutting_step'''
    fig, ax1 = plt.subplots()
    ax1.plot(cutting_steps, [data[step]['n_synapses'] for step in cutting_steps], 'ko--')
    ax1.set_ylim([0, 1.15*data[0]['n_synapses']])
    ax1.set_ylabel('Number of synapses', color='k')
    ax1.set_xlabel('Cutting step')

    # Annotate
    for step in cutting_steps:
        if step % 5 == 0 or step < 3:
            plt.annotate(str(data[step]['net_structure']), xy=(step, data[step]['n_synapses']+0.1*data[0]['n_synapses']), horizontalalignment='center', verticalalignment='center', fontsize=10, color='r')

    ax2 = ax1.twinx()
    ax2.plot(cutting_steps, [data[step]['accuracy'] for step in cutting_steps], 'go--')
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('Accuracy', color='g')
    for tl in ax2.get_yticklabels():
        tl.set_color('g')

    plt.xlim([-1, len(cutting_steps)+1])
    plt.title('Number of synapses and Accuracy vs. Cutting step : MNIST Dataset, 50000/10000 samples')
    red_patch = mpatches.Patch(color='red', label='Net structure')
    plt.legend(handles=[red_patch])
    plt.grid()
    plt.show()


    ''' n_synapses and epochs_needed vs. cutting_step'''
    fig, ax1 = plt.subplots()
    bar_width = 1.0
    for step in cutting_steps:
        ax1.bar(step-bar_width/2, sum([data[a_step]['plus_epochs_needed'] for a_step in range(1, step+1)]), bar_width, color='b')
    ax1.set_ylabel('Accumulated epochs needed', color='b')
    ax1.set_xlabel('Cutting step')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    ax2 = ax1.twinx()
    ax2.plot(cutting_steps, [data[step]['n_synapses'] for step in cutting_steps], 'ko--')
    ax2.set_ylim([0, 1.15*data[0]['n_synapses']])
    ax2.set_ylabel('Number of synapses', color='k')
    for tl in ax2.get_yticklabels():
        tl.set_color('k')

    # Annotate
    for step in cutting_steps:
        if step % 5 == 0 or step < 3:
            plt.annotate(str(data[step]['net_structure']), xy=(step, data[step]['n_synapses']+0.1*data[0]['n_synapses']), horizontalalignment='center', verticalalignment='center', fontsize=10, color='r')

    plt.xlim([-1, len(cutting_steps)+1])
    plt.title('Number of synapses and Epochs needed vs. Cutting step : MNIST Dataset, 50000/10000 samples')
    red_patch = mpatches.Patch(color='red', label='Net structure')
    plt.legend(handles=[red_patch])
    plt.grid()
    plt.show()"""


if __name__ == '__main__':
    plot_results_for_report(data=pickle.load(open('pickle/mnist_784-15-10_test2.p', 'rb')))