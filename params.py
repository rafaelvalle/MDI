#!/usr/bin/python

import os
import numpy as np
from scipy.stats import mode
import lasagne

# imputation parameters
params_dict = {}
imp_methods = ('RandomReplace', 'Summary', 'RandomForest', 'LogisticRegression',
               'SVM', 'KNN', 'PCA')
params_dict['miss_data_symbol'] = '?'
params_dict['miss_data_cond'] = lambda x: x == params_dict['miss_data_symbol']
params_dict['cat_cols'] = (1, 3, 4, 5, 6, 7, 8, 12) # Adult
params_dict['non_cat_cols'] = (0, 2, 9, 10, 11)
# params_dict['cat_cols'] = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16) # votes
# params_dict['non_cat_cols'] = ()
params_dict['n_neighbors'] = 3 # keep this <5 for votes
params_dict['summary_func'] = lambda x: mode(x)[0]
params_dict['knn_summary_func'] = np.mean

# folder paths
feats_train_folder = "data/train/features/"
labels_train_folder = "data/train/labels/"
results_train_folder = "results/train"
feats_test_folder = "data/test/features/"
labels_test_folder = "data/test/labels/"
results_test_folder = "results/test"
perturb_folder = "data/perturbed/"
scalers_folder = "data/scalers/"
imputed_folder = "data/imputed"
IMAGES_DIRECTORY = "images/"
RESULTS_PATH = 'results/'
TRIAL_DIRECTORY = os.path.join(RESULTS_PATH, 'parameter_trials')
MODEL_DIRECTORY = os.path.join(RESULTS_PATH, 'model')

# neural network parameter not to be explored with bayesian parameter estimation
nnet_params = {'n_folds': 1,
               'n_layers': 4,
               'batch_size': 16,
               'epoch_size': 128,
               'gammas': np.array([0.1, 0.01], dtype=np.float32),
               'decay_rate': 0.95,
               'max_epoch': 50,
               'widths': [None, 1024, 1024, 2],
               'non_linearities': (None,
                                   lasagne.nonlinearities.rectify,
                                   lasagne.nonlinearities.rectify,
                                   lasagne.nonlinearities.softmax),
               'update_func': lasagne.updates.adadelta,
               'drops': (None, 0.2, 0.5, None)}

# random number seed
rand_num_seed = 1

# hyperparameter space to be explored using bayesian parameter optimization
hyperparameter_space = {
    'momentum': {'type': 'float', 'min': 0., 'max': 1.},
    'dropout': {'type': 'int', 'min': 0, 'max': 1},
    'learning_rate': {'type': 'float', 'min': .000001, 'max': .01},
    'network': {'type': 'enum', 'options': ['general_network']}
    }
