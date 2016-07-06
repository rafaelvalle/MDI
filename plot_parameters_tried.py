#!/usr/local/bin/python

import os
import argparse
import glob
import numpy as np
import deepdish as dd
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
# import seaborn as sbn
from mpl_toolkits.mplot3d import Axes3D
from params import IMAGES_DIRECTORY

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def plot_3d(params_dir):
    N = 2 # bins for colormap
    model_dirs = [name for name in os.listdir(params_dir)
                  if os.path.isdir(os.path.join(params_dir, name))]

    colors = plt.get_cmap('plasma')
    plt.figure(figsize=(20, 10))
    ax = plt.subplot(111, projection='3d')
    ax.set_xlabel('Momentum')
    ax.set_ylabel('Learning Rate')
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel('Training error rate', rotation=270)
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.set_yticks(np.arange(0, 0.011, 0.002))
    ax.set_zticks(np.arange(0, 0.9, 0.1))
    #ax.set_xticklabels(('No', 'Yes'))
    #ax.set_zticklabels(('0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8'))
    ax.invert_yaxis() # invert y axis
    ax.invert_xaxis() # invert x axis
    #ax.view_init(azim=-178, elev=32)

    i = 0
    for model_dir in model_dirs:
        model_df = pd.DataFrame()
        for param_path in glob.glob(os.path.join(params_dir,
                                                 model_dir) + '/*.h5'):
            param = dd.io.load(param_path)
            gd = {'learning rate': param['hyperparameters']['learning_rate'],
                  'momentum': param['hyperparameters']['momentum'],
                  'dropout': param['hyperparameters']['dropout'],
                  'val. objective': param['best_epoch']['validate_objective']}
            model_df = model_df.append(pd.DataFrame(gd, index=[0]),
                                       ignore_index=True)
        if i != len(model_dirs) - 1:
            ax.scatter(model_df['momentum'],
                       model_df['learning rate'],
                       model_df['val. objective'],
                       s=128,
                       marker=(i+3, 0),
                       label=model_dir,
                   #    c=model_df['val. objective'],
                       c=model_df['dropout'],
                       cmap=discrete_cmap(N, 'jet'))
        else:
            im = ax.scatter(model_df['momentum'],
                            model_df['learning rate'],
                            model_df['val. objective'],
                            s=128,
                            marker=(i+4, 0),
                            label=model_dir,
                       #    c=model_df['val. objective'],
                            c=model_df['dropout'],
                            cmap=discrete_cmap(N, 'jet'))
        i += 1

    cbar=plt.colorbar(im, label='Dropout',ticks=range(N))
    cbar.ax.set_yticklabels(['No','Yes'])
    cbar.set_label('Dropout', rotation=270)
    #plt.legend()
    plt.title('Adult dataset',weight='bold')
    plt.show()
    plt.savefig('{}.eps'.format(os.path.join(IMAGES_DIRECTORY, 'params3d_adult')), format='eps', dpi=1000)
    plt.close()

def plot_2d(params_dir):
    model_dirs = [name for name in os.listdir(params_dir)
                  if os.path.isdir(os.path.join(params_dir, name))]
    if len(model_dirs) == 0:
      model_dirs = [params_dir]


    colors = plt.get_cmap('plasma')
    plt.figure(figsize=(20, 10))
    ax = plt.subplot(111)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Error rate')

    i = 0
    for model_dir in model_dirs:
        model_df = pd.DataFrame()
        for param_path in glob.glob(os.path.join(params_dir,
                                                 model_dir) + '/*.h5'):
            param = dd.io.load(param_path)
            gd = {'learning rate': param['hyperparameters']['learning_rate'],
                  'momentum': param['hyperparameters']['momentum'],
                  'dropout': param['hyperparameters']['dropout'],
                  'val. objective': param['best_epoch']['validate_objective']}
            model_df = model_df.append(pd.DataFrame(gd, index=[0]),
                                       ignore_index=True)
        if i != len(model_dirs) - 1:
            ax.scatter(model_df['learning rate'],
                       model_df['val. objective'],
                       s=128,
                       marker=(i+3, 0),
                       edgecolor='black',
                       linewidth=model_df['dropout'],
                       label=model_dir,
                       c=model_df['momentum'],
                       cmap=colors)
        else:
            im = ax.scatter(model_df['learning rate'],
                            model_df['val. objective'],
                            s=128,
                            marker=(i+3, 0),
                            edgecolor='black',
                            linewidth=model_df['dropout'],
                            label=model_dir,
                            c=model_df['momentum'],
                            cmap=colors)
        i += 1

    plt.colorbar(im, label='Momentum')
    plt.legend()
    plt.show()
    plt.savefig('{}.eps'.format(os.path.join(IMAGES_DIRECTORY, 'params2d')), format='eps', dpi=1000)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("params_dir", type=str,
                        help="Fullpath to parameter trial folders")

    parser.add_argument("ndims", type=int, default=2,
                        help="Fullpath to parameter trial folders")

    args = parser.parse_args()
    if args.ndims == 2:
        plot_2d(args.params_dir)
    elif args.ndims == 3:
        plot_3d(args.params_dir)
    else:
        raise Exception(
            "{} is not a valid number of dimensions".format(args.ndmins))
