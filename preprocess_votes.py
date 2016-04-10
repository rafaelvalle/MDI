import numpy as np
import pickle
import cPickle as pkl
import os, random
from scipy import delete
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from missing_data_imputation import Imputer
from processing import impute, perturbate_data
from params import imp_methods, params_dict, scalers_folder
from params import feats_train_folder, labels_train_folder, perturb_folder
from params import feats_test_folder, labels_test_folder
from params import rand_num_seed


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

np.random.seed(rand_num_seed)
random.seed(rand_num_seed)

# load features and labels
votes = np.genfromtxt('data/house-votes-84.data', delimiter=',', dtype=object)

# split data to (2/3) training and (1/3) test
votes_train,votes_test = train_test_split(votes, test_size=0.33)

# binarize labels
labels_train = (np.array(votes_train [:, 0]) == 'democrat').astype(int)
labels_train = labels_train.reshape((-1, 1))

labels_test = (np.array(votes_test [:, 0]) == 'democrat').astype(int)
labels_test = labels_test.reshape((-1, 1))

# save train labels in binary and one-hot representations
labels_train.dump(os.path.join(labels_train_folder, 'labels_bin.np'))
(np.eye(2)[labels_train]).astype(int).dump(os.path.join(labels_train_folder,
                                                  'labels_bin_onehot.np'))

# save test labels in binary and one-hot representations
labels_test.dump(os.path.join(labels_test_folder, 'labels_bin_test.np'))
(np.eye(2)[labels_test.astype(int)]).astype(int).dump(
    os.path.join(labels_test_folder, 'labels_onehot_test.np'))

## For training data 

# enumerate parameters
monotone = True
ratios = np.arange(.1, .5, .1)

for ratio in ratios:
    print '\nPerturbing {}% of data'.format(ratio)
    pert_data, _ = perturbate_data(votes_train, params_dict['cat_cols'], ratio, monotone,
                                   params_dict['miss_data_symbol'])

    path = os.path.join(perturb_folder,
                        'votes_train_pert_mono_{}_ratio_{}.csv'.format(monotone,
                                                                       ratio))
    # save perturbed data to disk as csv
    print '\tSaving perturbed data to {}'.format(path)
    np.savetxt(path, pert_data, delimiter=",", fmt="%s")

    for imp_method in imp_methods:
        print '\tImputing with {}'.format(imp_method)
        imp = Imputer()
        data = impute(pert_data, imp, imp_method, params_dict)

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
        # scaler = scaler.fit(data[:, params_dict['non_cat_cols']].astype(float))

        data_scaled = np.copy(data)
        # data_scaled[:, params_dict['non_cat_cols']] = scaler.transform(
        #     data[:, params_dict['non_cat_cols']].astype(float))

        # key is imputation method and ratio dependent
        # filename is imputation method dependent
        scaler_dict["{}_ratio_{}".format(imp_method, ratio)] = scaler
        pkl.dump(scaler_dict, open(scaler_path, 'wb'))


        # binarize scaled data
        data_scaled_bin = imp.binarize_data(data_scaled,
                                            params_dict['cat_cols'],
                                            params_dict['miss_data_symbol'])
        # convert to float
        data_scaled_bin = data_scaled_bin.astype(float)

        # add labels as last column
        data_scaled_bin = np.hstack((data_scaled_bin, labels_train))

        # save to disk
        filename = "{}_bin_scaled_mono_{}_ratio_{}.np".format(imp_method,
                                                              monotone,
                                                              ratio)
        path = os.path.join(feats_train_folder, filename)
        print '\tSaving imputed scaled and binarized data to {}'.format(path)
        data_scaled_bin.dump(path)

## For test data

# instantiate Imputer
imp = Imputer()

for imp_method in imp_methods:
    print 'Imputing with {}'.format(imp_method)
    data = impute(votes_test, imp, imp_method, params_dict)

    # load respective scaler
    scaler_path = os.path.join(scalers_folder,
                               "{}_scaler".format(imp_method))

    scaler_dict = pickle.load(open(scaler_path, "rb"))
    for name, scaler in scaler_dict.items():
        # scale and binarize, adding one col for missing value in all categ vars
        data_scaled = np.copy(data)
        # data_scaled[:, params_dict['non_cat_cols']] = scaler.transform(
        #     data[:, params_dict['non_cat_cols']].astype(float))
        data_scaled_bin = imp.binarize_data(data_scaled,
                                            params_dict['cat_cols'],
                                            params_dict['miss_data_symbol'])
        # convert to float
        data_scaled_bin = data_scaled_bin.astype(float)

        # add labels as last column
        path = os.path.join(feats_test_folder,
                            '{}_bin_scaled_test.np'.format(name))
        data_scaled_bin = np.hstack((data_scaled_bin, labels_test))
        print "\tSaving imputed data to {}".format(path)
        data_scaled_bin.dump(path)
    del data
    del data_scaled
    del data_scaled_bin