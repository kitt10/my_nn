
def cut_synapses(cut_level):


def converges(learning_rate):



net = Net([2, 15, 1])     # must be possible to learn to required success rate (0.9)
learning_rate = 'optimal for this problem'
learn_to_0.9(net, learning_rate)
cut_level = 1 # 1 : rough, ... 3 : soft
found = False

while not found:
    net_tmp = net
    net_tmp.cut_synapses(cut_level)
    if net_tmp.converges(learning_rate):
        net = net_tmp
    else:
        cut_level += 1

    if cut_level out of bounds:
        found = True

