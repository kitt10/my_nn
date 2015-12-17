__author__ = 'kitt'

import matplotlib.pyplot as plt


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

def plot_results_for_report_xor(data):
    pass