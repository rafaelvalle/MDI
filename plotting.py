import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

# helper function to plot histograms
def plot_histogram(freq_data, labels, axes, axis, width, title,
                   color_mapping):
    n_methods = len(freq_data.keys())
    labels = sorted(freq_data.values()[0].keys())
    bins = np.arange(len(labels))

    for i in xrange(n_methods):
        key = sorted(freq_data.keys())[i]
        offset = i*2*width/float(n_methods)
        try:
            values = [freq_data[key][label] for label in labels]
        except:
            set_trace()
        axes.flat[axis].bar(bins+offset, values,
                            width, label=key,
                            color=plt.cm.Set1(color_mapping[key]),
                            align='center')

    axes.flat[axis].set_xlim(bins[0]-0.5, bins[-1]+width+0.5)
    axes.flat[axis].set_title(title)
    axes.flat[axis].set_xticks(bins + width)
    axes.flat[axis].set_xticklabels(labels, rotation=90, fontsize='small')
    axes.flat[axis].legend(loc='best', prop={'size': 8},
                           shadow=True, fancybox=True)


def plot_confusion_matrix(y, y_predict, axes, axis, title='',
                          normalize=True, add_text=False):
    """Plots a confusion matrix given labels and predicted labels

    Parameters
    ----------
        y: ground truth labels <int array>
        y_predict: predicted labels <int array>
    """

    conf_mat = confusion_matrix(y, y_predict)
    if normalize:
        conf_mat_norm = conf_mat / \
                        conf_mat.sum(axis=1).astype(float)[:, np.newaxis]
        conf_mat_norm = np.nan_to_num(conf_mat_norm)

    axes.flat[axis].imshow(conf_mat_norm, cmap=plt.cm.Blues,
                           interpolation='nearest')
    if axis < axes.shape[1]:
        axes.flat[axis].set_title(title)

    # add text to confusion matrix
    if add_text:
        for x in xrange(conf_mat.shape[0]):
            for y in xrange(conf_mat.shape[0]):
                if conf_mat[x, y] > 0:
                    axes.flat[axis].annotate(str(conf_mat[x, y]), xy=(y, x),
                                             horizontalalignment='center',
                                             verticalalignment='center')
