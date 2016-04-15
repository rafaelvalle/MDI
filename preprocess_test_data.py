#!/usr/bin/python

import numpy as np
import cPickle as pickle
import os, random
from scipy import delete
from missing_data_imputation import Imputer
from processing import impute
from params import imp_methods, params_dict, scalers_folder
from params import feats_test_folder, labels_test_folder
from params import rand_num_seed


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

np.random.seed(rand_num_seed)
random.seed(rand_num_seed)

# load features and labels
x = np.genfromtxt('data/adult-test-raw', delimiter=', ', dtype=object)

# binarize labels
labels = (np.array(x[:, -1]) == '>50K').astype(int)
labels = labels.reshape((-1, 1))

# save labels in binary and one-hot representations
labels.dump(os.path.join(labels_test_folder, 'labels_bin_test.np'))
(np.eye(2)[labels.astype(int)]).astype(int).dump(
    os.path.join(labels_test_folder, 'labels_onehot_test.np'))

# remove redundant education-number and labels features
x = delete(x, (4, 14), 1)

# instantiate Imputer
imp = Imputer()

for imp_method in imp_methods:
    print 'Imputing with {}'.format(imp_method)
    data = impute(x, imp, imp_method, params_dict)

    # load respective scaler
    scaler_path = os.path.join(scalers_folder,
                               "{}_scaler".format(imp_method))

    scaler_dict = pickle.load(open(scaler_path, "rb"))
    for name, scaler in scaler_dict.items():
        # scale and binarize, adding one col for missing value in all categ vars
        data_scaled = np.copy(data)
        data_scaled[:, params_dict['non_cat_cols']] = scaler.transform(
            data[:, params_dict['non_cat_cols']].astype(float))
        data_scaled_bin = imp.binarize_data(data_scaled,
                                            params_dict['cat_cols'],
                                            params_dict['miss_data_symbol'])
        # convert to float
        data_scaled_bin = data_scaled_bin.astype(float)

        # add labels as last column
        path = os.path.join(feats_test_folder,
                            '{}_bin_scaled_test.np'.format(name))
        data_scaled_bin = np.hstack((data_scaled_bin, labels))
        print "\tSaving imputed data to {}".format(path)
        data_scaled_bin.dump(path)
    del data
    del data_scaled
    del data_scaled_bin
