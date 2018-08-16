import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
#%matplotlib inline

def plot_loss(loss_list, label, path):
    plt.plot(*zip(*loss_list), label=label)
    plt.legend(loc='upper left')
    #plt.show()
    plt.savefig(path)
    plt.close()
