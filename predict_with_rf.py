#!/usr/bin/python

"""This function loads the best models trained so far and use them to make
predictions using the datasets in the given include file"""

import os
import argparse
import cPickle as pkl
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from params import feats_train_folder, feats_test_folder
from params import MODEL_DIRECTORY, RESULTS_PATH


dataname = 'votes'


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


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

    # store predictions on a dictionary
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

            # Fit Tree Classifiers
            clfs = {'DTC(max_depth=None)':
                DecisionTreeClassifier(max_depth=None),
            'DTC(max_depth=5)':
                DecisionTreeClassifier(max_depth=5),
            'DTC(max_depth=10)':
                DecisionTreeClassifier(max_depth=10),
            'DTC(max_depth=20)':
                DecisionTreeClassifier(max_depth=20),
            'DTC(max_depth=25)':
                DecisionTreeClassifier(max_depth=25),
            'DTC(max_depth=50)':
                DecisionTreeClassifier(max_depth=50),
            'DTC(max_depth=100)':
                DecisionTreeClassifier(max_depth=100),
            'DTC(max_depth=500)':
                DecisionTreeClassifier(max_depth=500),
            'DTC(max_depth=1000)':
                DecisionTreeClassifier(max_depth=1000),
            'DTC(max_depth=5000)':
                DecisionTreeClassifier(max_depth=5000),
            'RFC(n_estimators=10, max_features="sqrt")':
                RandomForestClassifier(n_estimators=10, max_features="sqrt"),
            'RFC(n_estimators=50, max_features="log2")':
                RandomForestClassifier(n_estimators=50, max_features="log2"),
            'RFC(n_estimators=100, max_features=None)':
                RandomForestClassifier(n_estimators=100, max_features=None),
            'RFC(n_estimators=500, max_features="sqrt")':
                RandomForestClassifier(n_estimators=500, max_features="sqrt"),
            'RFC(n_estimators=1000, max_features="log2")':
                RandomForestClassifier(n_estimators=1000, max_features="log2"),
            'RFC(n_estimators=1500, max_features=None)':
                RandomForestClassifier(n_estimators=1500, max_features=None),
            'RFC(n_estimators=2000, max_features="sqrt")':
                RandomForestClassifier(n_estimators=2000, max_features="sqrt"),
            'RFC(n_estimators=2500, max_features="log2")':
                RandomForestClassifier(n_estimators=2500, max_features="log2"),
            'RFC(n_estimators=3000, max_features=None)':
                RandomForestClassifier(n_estimators=3000, max_features=None),
            'RFC(n_estimators=3500, max_features="sqrt")':
                RandomForestClassifier(n_estimators=3500, max_features="sqrt")}

            for model_name, clf in clfs.items():
                clf.fit(train_data[:,:-1], train_data[:, -1].astype(int))
                y_test_hat = clf.predict(test_data[:,:-1])
                obj_val = (sum(y_test_hat != test_data[:, -1]) /
                    float(len(test_data)))

                model_preds[model_name+imputation_name] = obj_val
                print("{} on {} error rate on test set: {}").format(
                    model_name, imputation_name, obj_val)

    # dump dictionary
    pkl.dump(model_preds, open(
        os.path.join(RESULTS_PATH, 'trees_{}_results.np'.format(dataname)), 
        'wb'))

    # print dictionary
    dumpclean(model_preds)