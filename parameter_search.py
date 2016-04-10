""" Search for good hyperparameters for classifiction using the manually
perturbed (missing data) ADULT dataset.
"""

import os
import numpy as np
import neural_networks
import bayesian_parameter_optimization as bpo
from params import nnet_params, hyperparameter_space, feats_train_folder

RESULTS_PATH = 'results/'


if __name__ == '__main__':
    # Construct paths
    trial_directory = os.path.join(RESULTS_PATH, 'parameter_trials')
    model_directory = os.path.join(RESULTS_PATH, 'model')

    # train on every perturbed dataset
    filepaths = np.loadtxt("include_data.csv", dtype=object, delimiter=",")
    for (include, train_filename, test_filename) in filepaths:
        if include == '1':
            print ('\nExecuting bayesian parameter optimization'
                   '\n{}').format(train_filename)
            # Load training and validation sets
            data = np.load(os.path.join(feats_train_folder,
                                        train_filename)).astype(np.float32)
    # Run parameter optimization forever
    bpo.parameter_search(data,
                         nnet_params,
                         hyperparameter_space,
                         trial_directory,
                         model_directory,
                         neural_networks.train)
