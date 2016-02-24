import numpy as np
from scipy.stats import mode

# imputation parameters
params_dict = {}
# imp_methods = ['RandomReplace', 'Summary', 'RandomForest', 'PCA', 'KNN']
imp_methods = ['RandomReplace', 'Summary', 'RandomForest', 'PCA']
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
feats_test_folder = "data/test/features/"
labels_test_folder = "data/test/labels/"
perturb_folder = "data/perturbed/"
scalers_folder = "data/scalers/"

# neural network parameters
nnet_params_dict = {}
nnet_params_dict['n_folds'] = 3
nnet_params_dict['alphas'] = (1, 4, 9)
nnet_params_dict['batch_sizes'] = (32, 512, 4096)
nnet_params_dict['gammas'] = np.array([ 0.1 ,  0.01])
nnet_params_dict['max_epoch'] = 50