import numpy as np
from scipy.stats import mode
import lasagne

# imputation parameters
params_dict = {}
imp_methods = ('RandomReplace', 'Summary', 'RandomForest', 'PCA')
# imp_methods = ('KNN',)
params_dict['miss_data_symbol'] = '?'
params_dict['miss_data_cond'] = lambda x: x == params_dict['miss_data_symbol']
params_dict['cat_cols'] = (1, 3, 4, 5, 6, 7, 8, 12)
params_dict['non_cat_cols'] = (0, 2, 9, 10, 11)
params_dict['n_neighbors'] = 5
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

# neural network parameters
nnet_params_dict = {}
nnet_params_dict['n_folds'] = 3
nnet_params_dict['depth'] = 3
nnet_params_dict['alphas'] = np.array([0.1, 1, 4], dtype=np.float32)
nnet_params_dict['batch_sizes'] = (64, 512, 4096)
nnet_params_dict['gammas'] = np.array([0.1, 0.01], dtype=np.float32)
nnet_params_dict['decay_rate'] = 0.95
nnet_params_dict['max_epoch'] = 50
nnet_params_dict['widths'] = (0, 1000, 2)
nnet_params_dict['nonlins'] = (None, lasagne.nonlinearities.rectify,
lasagne.nonlinearities.softmax)
nnet_params_dict['drops'] = (0.2, 0.5, None)

# random number seed
rand_num_seed = 1
