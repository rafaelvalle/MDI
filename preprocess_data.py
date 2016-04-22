#!/usr/bin/python

import numpy as np
import cPickle as pkl
import os
import random
from scipy import delete
from sklearn.preprocessing import StandardScaler
from missing_data_imputation import Imputer
from processing import impute, perturbate_data
from params import imp_methods, adult_params, scalers_folder
from params import feats_train_folder, labels_train_folder, perturb_folder
from params import rand_num_seed


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

np.random.seed(rand_num_seed)
random.seed(rand_num_seed)

# load features and labels
x = np.genfromtxt('data/adult-train-raw', delimiter=', ', dtype=object)

# remove holland from data to match feature size in test data
x = x[x[:, -2] != 'Holand-Netherlands']

# binarize labels
labels = (np.array(x[:, -1]) == '>50K').astype(int)
labels = labels.reshape((-1, 1))

# save labels in binary and one-hot representations
labels.dump(os.path.join(labels_train_folder, 'labels_bin.np'))

# remove redundant education-number and labels features
x = delete(x, (4, 14), 1)

# enumerate parameters
monotone = True
ratios = np.arange(0, .5, .1)

for ratio in ratios:
    print '\nPerturbing {}% of data'.format(ratio)
    pert_data, _ = perturbate_data(x, adult_params['cat_cols'], ratio, monotone,
                                   adult_params['miss_data_symbol'])
    print "\tRatio is {} of {}".format(
            np.sum(pert_data == adult_params['miss_data_symbol']), 
            len(pert_data) * len(adult_params['cat_cols']))

    path = os.path.join(perturb_folder,
                        'adult_train_pert_mono_{}_ratio_{}.csv'.format(monotone,
                                                                       ratio))
    # save perturbed data to disk as csv
    print '\tSaving perturbed data to {}'.format(path)
    np.savetxt(path, pert_data, delimiter=",", fmt="%s")

    for imp_method in imp_methods:
        print '\tImputing with {}'.format(imp_method)
        imp = Imputer()
        data = impute(pert_data, imp, imp_method, adult_params)

        path = "data/imputed/{}_mono_{}_ratio_{}.csv".format(imp_method,
                                                             monotone,
                                                             ratio)
        # save data as csv
        print '\tSaving imputed data to {}'.format(path)
        np.savetxt(path, data, delimiter=",", fmt="%s")

        # scale continuous variables and convert categorial to one-hot
        # store the scaler objects to be used on the test set
        scaler_path = os.path.join(scalers_folder,
                                   "{}_scaler".format(imp_method))

        if os.path.isfile(scaler_path):
            scaler_dict = pkl.load(open(scaler_path, "rb"))
        else:
            scaler_dict = {}

        scaler = StandardScaler()
        scaler = scaler.fit(data[:, adult_params['non_cat_cols']].astype(float))

        data_scaled = np.copy(data)
        data_scaled[:, adult_params['non_cat_cols']] = scaler.transform(
            data[:, adult_params['non_cat_cols']].astype(float))

        # key is imputation method and ratio dependent
        # filename is imputation method dependent
        scaler_dict["{}_ratio_{}".format(imp_method, ratio)] = scaler
        pkl.dump(scaler_dict, open(scaler_path, 'wb'))

        # binarize scaled data
        data_scaled_bin = imp.binarize_data(data_scaled,
                                            adult_params['cat_cols'],
                                            adult_params['miss_data_symbol'])
        # convert to float
        data_scaled_bin = data_scaled_bin.astype(float)

        # add labels as last column
        data_scaled_bin = np.hstack((data_scaled_bin, labels))

        # save to disk
        filename = "{}_bin_scaled_mono_{}_ratio_{}.np".format(imp_method,
                                                              monotone,
                                                              ratio)
        path = os.path.join(feats_train_folder, filename)
        print '\tSaving imputed scaled and binarized data to {}'.format(path)
        data_scaled_bin.dump(path)
