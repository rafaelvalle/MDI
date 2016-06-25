#!/usr/local/bin/python
from collections import defaultdict
import os
import re
import argparse
import glob
import deepdish as dd
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import seaborn as sns
from params import IMAGES_DIRECTORY


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


def plot(params_dir):
    model_dirs = [name for name in os.listdir(params_dir)
                  if os.path.isdir(os.path.join(params_dir, name))]

    df = defaultdict(list)
    for model_dir in model_dirs:
        df[re.sub('_bin_scaled_mono_True_ratio', '', model_dir)] = [
            dd.io.load(path)['best_epoch']['validate_objective']
            for path in glob.glob(os.path.join(
                params_dir, model_dir) + '/*.h5')]

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in df.iteritems()]))
    plt.figure(figsize=(16, 4), dpi=300)
    g = sns.boxplot(df)
    g.set_xticklabels(df.columns, rotation=45)
    plt.tight_layout()
    plt.savefig('{}_errors_box_plot.png'.format(
        os.path.join(IMAGES_DIRECTORY,
                     os.path.basename(os.path.normpath(params_dir)))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("params_dir", type=str,
                        help="Fullpath to parameter trial folders")

    args = parser.parse_args()
    plot(args.params_dir)
