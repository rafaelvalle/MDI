#!/usr/bin/python

"""This function loads the best models trained so far and use them to make
predictions using the datasets in the given include file"""

import os
import argparse
import cPickle as pkl
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from params import feats_train_folder, feats_test_folder
from params import RESULTS_PATH
import pdb


dataname = 'votes'


def dumpclean(obj):
    if type(obj) == dict:
        for k, v in obj.items():
            if hasattr(v, '__iter__'):
                print k
                dumpclean(v)
            else:
                print '%s : %s' % (k, v)
    elif type(obj) == list:
        for v in obj:
            if hasattr(v, '__iter__'):
                dumpclean(v)
            else:
                print v
    else:
        print obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "include_path", type=str,
        help="Path to CSV file with rows as 'include ?, train_file, test_file'")
    args = parser.parse_args()

    # store predictions in a dictionary
    model_preds = {}
    filepaths = np.loadtxt(args.include_path, dtype=object, delimiter=",")
    for (include, train_path, test_path) in filepaths:
        if include == '1':
            imputation_name = os.path.basename(train_path)[:-3]
            print("\nExecuting prediction on "
                  "test set\n{}").format(imputation_name)
            # Load train and test set
            train_data = np.load(
                os.path.join(feats_train_folder, train_path)).astype(np.float32)
            test_data = np.load(
                os.path.join(feats_test_folder, test_path)).astype(np.float32)
            # define parameters for DT and RF classifiers
            dtc_parameters = dict(criterion=('gini', 'entropy'),
                                  splitter=('best', 'random'),
                                  max_features=('sqrt', 'log2', None),
                                  max_depth=[2**i for i in xrange(1, 10)],
                                  class_weight=['balanced']
                                  )
            rfc_parameters = dict(n_estimators=[2**i for i in xrange(1, 10)],
                                  criterion=('gini', 'entropy'),
                                  max_features=('sqrt', 'log2', None),
                                  class_weight=['balanced']
                                  )
            # train DT and RF models using grid search cross validation
            dtc = GridSearchCV(DecisionTreeClassifier(), dtc_parameters)
            rfc = GridSearchCV(RandomForestClassifier(), rfc_parameters)
            dtc.fit(train_data[:, :-1], train_data[:, -1].astype(int))
            rfc.fit(train_data[:, :-1], train_data[:, -1].astype(int))

            # predict with cv'ed models
            dt_y_test_hat = dtc.predict(test_data[:, :-1])
            rf_y_test_hat = rfc.predict(test_data[:, :-1])

            # compute objective function
            dt_obj_val = (sum(dt_y_test_hat != test_data[:, -1]) /
                          float(len(test_data)))
            rf_obj_val = (sum(rf_y_test_hat != test_data[:, -1]) /
                          float(len(test_data)))

            model_preds['DT'+imputation_name] = dt_obj_val
            model_preds['RF'+imputation_name] = rf_obj_val
            print("DT on {} error rate on test set: {}").format(
                imputation_name, dt_obj_val)
            print("RF on {} error rate on test set: {}").format(
                imputation_name, rf_obj_val)

    # dump dictionary
    pkl.dump(model_preds, open(
        os.path.join(RESULTS_PATH, 'trees_cved_{}_results.np'.format(dataname)),
        'wb'))

    # print dictionary
    dumpclean(model_preds)
