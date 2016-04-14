#!/usr/bin/python

"""This function loads the best models trained so far and use them to make
predictions using the datasets in the given include file"""
import os
import argparse
import cPickle as pkl
import numpy as np
import theano
from theano import tensor as T
import lasagne
import deepdish
from params import feats_test_folder, MODEL_DIRECTORY, RESULTS_PATH
from params import nnet_params, hyperparameter_space
import neural_networks


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


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
            model_name = os.path.basename(train_path)[:-3]
            print("\nExecuting prediction on test set \n{}").format(model_name)

            # Load test set, separate target labels from dataset
            data = np.load(
                os.path.join(feats_test_folder, test_path)).astype(np.float32)

            network = neural_networks.build_general_network(
                (nnet_params['batch_size'], data.shape[1]-1),  # last is target
                nnet_params['n_layers'],
                nnet_params['widths'],
                nnet_params['non_linearities'],
                drop_out=False)

            # load best network model so far
            parameters = deepdish.io.load(
                os.path.join(MODEL_DIRECTORY, model_name+'.h5'))
            lasagne.layers.set_all_param_values(network, parameters)

            # set up neural network functions for predictions
            input_var = T.fmatrix()
            target_var = T.ivector()
            prediction = lasagne.layers.get_output(
                network, input_var, deterministic=True)
            obj_fn = T.mean(T.neq(T.argmax(prediction, axis=1), target_var))
            validate_fn = theano.function(
                inputs=[input_var, target_var], outputs=[obj_fn])

            # compute predictions. last column is target variable
            set_trace()
            obj_val = validate_fn(data[:, :-1].astype(input_var.dtype),
                                  data[:, -1].astype(target_var.dtype))
            model_preds[model_name] = obj_val
            print("{} error rate on test set: {}").format(model_name, obj_val)

    # dump dictionary
    pkl.dump(model_preds, open(os.path.join(RESULTS_PATH, 'results.np', 'wb')))
