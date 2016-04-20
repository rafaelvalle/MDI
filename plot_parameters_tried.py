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


def plot_3d(params_dir):
    model_dirs = [name for name in os.listdir(params_dir)
                  if os.path.isdir(os.path.join(params_dir, name))]

    colors = plt.get_cmap('plasma')
    plt.figure(figsize=(20, 10))
    ax = plt.subplot(111, projection='3d')
    ax.set_xlabel('Dropout')
    ax.set_ylabel('Learning Rate')
    ax.set_zlabel('Momentum')
    ax.set_xticks((0, 1))
    ax.set_zticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels(('False', 'True'))
    ax.view_init(azim=-178, elev=32)

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
            ax.scatter(model_df['dropout'],
                       model_df['learning rate'],
                       model_df['momentum'],
                       s=128,
                       marker=(i+3, 0),
                       label=model_dir,
                       c=model_df['val. objective'],
                       cmap=colors)
        else:
            im = ax.scatter(model_df['dropout'],
                            model_df['learning rate'],
                            model_df['momentum'],
                            s=128,
                            marker=(i+4, 0),
                            label=model_dir,
                            c=model_df['val. objective'],
                            cmap=colors)
        i += 1

    plt.colorbar(im, label='Error rate')
    plt.legend()
    plt.show()
    plt.savefig('{}.png'.format(os.path.join(IMAGES_DIRECTORY, 'params3d')))
    plt.close()

def plot_2d(params_dir):
    model_dirs = [name for name in os.listdir(params_dir)
                  if os.path.isdir(os.path.join(params_dir, name))]

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
    plt.savefig('{}.png'.format(os.path.join(IMAGES_DIRECTORY, 'params2d')))
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
