#!/usr/bin/python

import os
import numpy as np
from scipy.stats import mode
import lasagne


# random number seed
rand_num_seed = 1

# imputation parameters

adult_params = {
    'miss_data_symbol': '?',
    'miss_data_cond': lambda x: x == '?',
    'cat_cols': (1, 3, 4, 5, 6, 7, 8, 12),
    'non_cat_cols': (0, 2, 9, 10, 11),
    'mnar_values': ('Without-pay', 'Never-worked',
      '1st-4th', '10th','5th-6th',"9th, 7th-8th", "11th","12th",'Preschool',
      'Divorced', 'Never-married', 'Separated', 'Widowed',
      'Handlers-cleaners', 'Machine-op-inspct', 'Farming-fishing', 'Priv-house-ser',
      'Not-in-family', 'Other-relative', 'Unmarried',
      'Amer-Indian-Eskimo', 'Other', 'Black',
      'Male',
      'Cambodia', 'Puerto-Rico', 'Outlying-US(Guam-USVI-etc)', 'India', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Jamaica', 'Vietnam', 'Mexico', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia',"Guatemala", "Nicaragua", "Scotland", "Thailand", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong"),
    'imp_methods':('RandomReplace', 'Summary', 'RandomForest', 'LogisticRegression',
               'SVM', 'KNN','PCA', 'Identity'),
    'n_neighbors': 5,
    'knn_summary_func': np.mean,
    'summary_func': lambda x: mode(x)[0],
}

votes_params = {
    'miss_data_symbol': '?',
    'miss_data_cond': lambda x: x == '?',
    'cat_cols': np.arange(0, 16), # labels are not included in imputation
    'non_cat_cols': (),
    'mnar_values': ('n'),
    'imp_methods':('RandomReplace', 'Summary', 'RandomForest', 'LogisticRegression', #no KNN
               'SVM', 'PCA', 'Identity'), 
    'summary_func': lambda x: mode(x)[0]
}

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

# hyperparameter space to be explored using bayesian parameter optimization
hyperparameter_space = {
    'momentum': {'type': 'float', 'min': 0., 'max': 1.},
    'dropout': {'type': 'int', 'min': 0, 'max': 1},
    'learning_rate': {'type': 'float', 'min': .000001, 'max': .01},
    'network': {'type': 'enum', 'options': ['general_network']}
    }
