#!/usr/bin/python

""" Search for good hyperparameters for classifiction using the manually
perturbed (missing data) ADULT and VOTES datasets.
"""
import os
import argparse
import numpy as np
import neural_networks
import bayesian_parameter_optimization as bpo
from params import nnet_params, hyperparameter_space, feats_train_folder
from params import TRIAL_DIRECTORY, MODEL_DIRECTORY


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "include_path", type=str,
        help="Path to CSV file with rows as 'include ?, train_file, test_file'")
    parser.add_argument(
        "dataset_index", type=int,
        help="Row index of datasets in include_path to be analyzed")
    parser.add_argument(
        "dataset", type=str,
        help="Dataset name (adult or votes")

    args = parser.parse_args()

    filepaths = np.loadtxt(args.include_path, dtype=object, delimiter=",")
    model_name = os.path.basename(filepaths[args.dataset_index, 1])[:-3]
    print("\nExecuting bayesian parameter optimization\n{}").format(model_name)

    # Load training and validation sets and convert them to float32
    data = np.load(
        os.path.join(feats_train_folder, filepaths[args.dataset_index, 1]))
    data = data.astype(np.float32)

    # Run parameter optimization FOREVER
    bpo.parameter_search(data,
                         nnet_params,
                         hyperparameter_space,
                         os.path.join(TRIAL_DIRECTORY+"_"+args.dataset, model_name),
                         MODEL_DIRECTORY,
                         neural_networks.train,
                         model_name)
