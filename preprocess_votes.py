#!/usr/bin/python

import numpy as np
import os
import random
from scipy import delete
from sklearn.cross_validation import train_test_split
from missing_data_imputation import Imputer
from processing import impute, perturbate_data
from params import imp_methods, votes_params
from params import feats_train_folder, labels_train_folder, perturb_folder
from params import feats_test_folder, labels_test_folder
from params import rand_num_seed


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

np.random.seed(rand_num_seed)
random.seed(rand_num_seed)

dataname = 'votes'

# load features and labels
votes = np.genfromtxt('data/house-votes-84.data', delimiter=',', dtype=object)

# split data to (2/3) training and (1/3) test
votes_train, votes_test = train_test_split(votes, test_size=0.33)

# binarize labels
labels_train = (votes_train[:, 0] == 'democrat').astype(int)
labels_train = labels_train.reshape((-1, 1))
labels_test = (votes_test[:, 0] == 'democrat').astype(int)
labels_test = labels_test.reshape((-1, 1))

# save train labels in binary and one-hot representations
labels_train.dump(os.path.join(
    labels_train_folder, '{}_labels_bin.np'.format(dataname)))

# save test labels in binary and one-hot representations
labels_test.dump(os.path.join(
    labels_test_folder, '{}_labels_bin_test.np'.format(dataname)))

# remove labels column
votes_train = delete(votes_train, 0, 1)
votes_test = delete(votes_test, 0, 1)

# save votes training data
np.savetxt('data/votes_train.csv', votes_train, delimiter=",", fmt="%s")

# For training data
print 'Preparing train data for {}'.format(dataname)

# enumerate parameters
monotone = True
ratios = np.arange(0, .2, .1)

for ratio in ratios:
    print '\nPerturbing {}% of data'.format(ratio)
    pert_data, _ = perturbate_data(
        votes_train, votes_params['cat_cols'], ratio, monotone,
        votes_params['miss_data_symbol'])
    path = os.path.join(perturb_folder,
                        '{}_train_pert_mono_{}_ratio_{}.csv'.format(dataname,
                                                                    monotone,
                                                                    ratio))
    # save perturbed data to disk as csv
    print '\tSaving perturbed data to {}'.format(path)
    np.savetxt(path, pert_data, delimiter=",", fmt="%s")
    # impute data given imp_methods in params.py
    for imp_method in imp_methods:
        print '\tImputing with {}'.format(imp_method)
        imp = Imputer()
        data = impute(pert_data, imp, imp_method, votes_params)
        path = "data/imputed/{}_{}_mono_{}_ratio_{}.csv".format(dataname,
                                                                imp_method,
                                                                monotone,
                                                                ratio)
        # save data as csv
        print '\tSaving imputed data to {}'.format(path)
        np.savetxt(path, data, delimiter=",", fmt="%s")

        # binarize data
        data_scaled_bin = imp.binarize_data(data,
                                            votes_params['cat_cols'],
                                            votes_params['miss_data_symbol'])
        # convert to float
        data_scaled_bin = data_scaled_bin.astype(float)

        # add labels as last column
        data_scaled_bin = np.hstack((data_scaled_bin, labels_train))

        # save to disk
        filename = "{}_{}_bin_scaled_mono_{}_ratio_{}.np".format(dataname,
                                                                 imp_method,
                                                                 monotone,
                                                                 ratio)
        path = os.path.join(feats_train_folder, filename)
        print '\tSaving imputed scaled and binarized data to {}'.format(path)
        data_scaled_bin.dump(path)

# For test data
print 'Preparing test data for {}'.format(dataname)
# instantiate Imputer
imp = Imputer()
for imp_method in imp_methods:
    print 'Imputing with {}'.format(imp_method)
    data = impute(votes_test, imp, imp_method, votes_params)
    # scaling is not needed for votes data

    # scale and binarize, adding one col for missing value in all cat vars
    data_bin = np.copy(data)
    data_bin = imp.binarize_data(data_bin,
                                 votes_params['cat_cols'],
                                 votes_params['miss_data_symbol'])
    # convert to float
    data_bin = data_bin.astype(float)

    # add labels as last column
    path = os.path.join(feats_test_folder,
                        '{}_{}_bin_scaled_test.np'.format(dataname,
                                                          imp_method))
    data_bin = np.hstack((data_bin, labels_test))
    print "\tSaving imputed data to {}".format(path)
    data_bin.dump(path)
    del data
    del data_bin
