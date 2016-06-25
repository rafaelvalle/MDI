#!/usr/bin/python

"""This function loads the best models trained so far and use them to make
predictions using the datasets in the given include file"""

import os
import argparse
from collections import defaultdict
import cPickle as pkl
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from params import feats_train_folder, feats_test_folder
from params import MODEL_DIRECTORY, RESULTS_PATH
from params import adult_params
import matplotlib.pylab as plt
import seaborn
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

dataname = 'adult'


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
    parser.add_argument(
        "n_iterations", type=int,
        help="Number of iterations")

    args = parser.parse_args()

    # store predictions on a dictionary
    filepaths = np.loadtxt(args.include_path, dtype=object, delimiter=",")
    for (include, train_path, test_path) in filepaths:
        model_preds = defaultdict(list)
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
            clfs = {'DTC(max_depth=4)':
                        DecisionTreeClassifier(max_depth=4),
                    'DTC(max_depth=8)':
                        DecisionTreeClassifier(max_depth=8),
                    'DTC(max_depth=16)':
                        DecisionTreeClassifier(max_depth=16),
                    'DTC(max_depth=32)':
                        DecisionTreeClassifier(max_depth=32),
                    'DTC(max_depth=256)':
                        DecisionTreeClassifier(max_depth=256),
                    'DTC(max_depth=1024)':
                        DecisionTreeClassifier(max_depth=1024),
                    'RFC(max_depth=4)':
                        RandomForestClassifier(max_depth=4),
                    'RFC(max_depth=8)':
                        RandomForestClassifier(max_depth=8),
                    'RFC(max_depth=16)':
                        RandomForestClassifier(max_depth=16),
                    'RFC(max_depth=32)':
                        RandomForestClassifier(max_depth=32),
                    'RFC(max_depth=256)':
                        RandomForestClassifier(max_depth=256),
                    'RFC(max_depth=1024)':
                        RandomForestClassifier(max_depth=1024)}

            for _ in xrange(args.n_iterations):
                for model_name, clf in clfs.items():
                    clf.fit(train_data[:,:-1], train_data[:, -1].astype(int))
                    y_test_hat = clf.predict(test_data[:,:-1])
                    obj_val = (sum(y_test_hat != test_data[:, -1]) /
                        float(len(test_data)))

                    model_preds[model_name+imputation_name].append(obj_val)
                    print("{} on {} error rate on test set: {}").format(
                        model_name, imputation_name, obj_val)
            plt.figure(figsize=(30,20))
            plt.boxplot(model_preds.values(),
                        notch=True,
                        showmeans=True)
            plt.xticks(xrange(1, len(model_preds)+1),
                      model_preds.keys(),
                      rotation=60,
                      ha='right')
            plt.savefig("{}.png".format(imputation_name))
            # plt.tight_layout()
            plt.cla()

"""
    # predict using raw data
    train_data = np.genfromtxt(
        'data/adult-train-raw', delimiter=', ', dtype=object)
    test_data = np.genfromtxt(
        'data/adult-test-raw', delimiter=', ', dtype=object)
    # remove redundant education-number
    train_data = np.delete(train_data, 4, 1)
    test_data = np.delete(test_data, 4, 1)

    # create label encoders and replace data
    for col in adult_params['cat_cols']:
        encoder = LabelEncoder().fit(train_data[:, col])
        train_data[:, col] = encoder.transform(train_data[:, col])
        test_data[:, col] = encoder.transform(test_data[:, col])

    clfs = {
            'DTC(max_depth=4)':
                DecisionTreeClassifier(max_depth=4),
            'DTC(max_depth=8)':
                DecisionTreeClassifier(max_depth=8),
            'DTC(max_depth=16)':
                DecisionTreeClassifier(max_depth=16),
            'DTC(max_depth=32)':
                DecisionTreeClassifier(max_depth=32),
            'DTC(max_depth=256)':
                DecisionTreeClassifier(max_depth=256),
            'DTC(max_depth=1024)':
                DecisionTreeClassifier(max_depth=1024),
            'RFC(max_depth=4)':
                RandomForestClassifier(max_depth=4),
            'RFC(max_depth=8)':
                RandomForestClassifier(max_depth=8),
            'RFC(max_depth=16)':
                RandomForestClassifier(max_depth=16),
            'RFC(max_depth=32)':
                RandomForestClassifier(max_depth=32),
            'RFC(max_depth=256)':
                RandomForestClassifier(max_depth=256),
            'RFC(max_depth=1024)':
                RandomForestClassifier(max_depth=1024)}

    for model_name, clf in clfs.items():
        clf.fit(train_data[:,:-1], train_data[:, -1])
        y_test_hat = clf.predict(test_data[:,:-1])
        obj_val = (sum(y_test_hat != test_data[:, -1]) /
            float(len(test_data)))

        model_preds[model_name+'RawData'] = obj_val
        print("{} on {} error rate on test set: {}").format(
            model_name, 'Raw_Data', obj_val)
    # dump dictionary
    pkl.dump(model_preds, open(
        os.path.join(RESULTS_PATH, 'trees_{}_results.np'.format(dataname)),
        'wb'))

    # print dictionary
    dumpclean(model_preds)
    """