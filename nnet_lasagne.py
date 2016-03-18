# code adapted from lasagne tutorial
# http://lasagne.readthedocs.org/en/latest/user/tutorial.html

import time
import os
from itertools import product
import numpy as np
from sklearn.cross_validation import KFold
import theano
from theano import tensor as T
import lasagne
from params import nnet_params_dict, feats_train_folder


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


def build_network(input_var, shape, nonlins, depth=2,
                  widths=(1000, 1000, 10), drops=(0.2, 0.5)):
    """
    Parameters
    ----------
    input_var : Theano symbolic variable or None (default: None)
        Variable representing a  network input.
    shape : tuple of int or None (batchsize, rows, cols)
        Shape of the input. Any element can be set to None to indicate that
        dimension is not fixed at compile time

    """

    # GlorotUniform is the default mechanism for initializing weights
    for i in range(depth):
        if i == 0:
            network = lasagne.layers.InputLayer(shape=shape,
                                                input_var=input_var)
        else:
            network = lasagne.layers.DenseLayer(network,
                                                widths[i],
                                                nonlinearity=nonlins[i])
        if drops[i] != None:
            network = lasagne.layers.DropoutLayer(network, p=drops[i])

    return network


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def zerosX(X):
    return np.zeros(X, dtype=theano.config.floatX)


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))


def sgd(cost, params, gamma):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * gamma])
    return updates


def model(X, w_h, w_o):
    h = T.nnet.sigmoid(T.dot(X, w_h))
    pyx = T.nnet.softmax(T.dot(h, w_o))
    return pyx


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def batch_ids(batch_size, x_train, train_idx):
    # change to iterator
    ids = zip(range(0, len(x_train[train_idx]), batch_size),
              range(batch_size, len(x_train[train_idx]), batch_size))
    return ids

verbose = True
# train on every perturbed dataset
filepaths = np.loadtxt("include_data.csv", dtype=object, delimiter=",")
for (include, train_filename, test_filename) in filepaths:
    if include == '1':
        print '\nExecuting {}'.format(train_filename)
        # Load training and test sets
        x_train = np.load(os.path.join(feats_train_folder,
                                       train_filename)).astype(np.float32)

        y_train = x_train[:, -1].astype(int)
        # y_train = (np.eye(2, dtype=np.float32)[x_train[:,-1].astype(int)])

        # remove label column from x_train
        x_train = x_train[:, :-1]

        # Network topology
        n_obs = x_train.shape[0]
        n_inputs = x_train.shape[1]
        n_outputs = len(np.unique(y_train))

        # Cross-validation and Neural Net parameters
        n_folds = nnet_params_dict['n_folds']
        alphas = nnet_params_dict['alphas']
        gammas = nnet_params_dict['gammas']
        decay_rate = nnet_params_dict['decay_rate']
        batch_sizes = nnet_params_dict['batch_sizes']
        max_epoch = nnet_params_dict['max_epoch']
        depth = nnet_params_dict['depth']
        widths = nnet_params_dict['widths']
        nonlins = nnet_params_dict['nonlins']
        drops = nnet_params_dict['drops']

        # Dictionary to store results
        results_dict = {}

        params_mat = [x for x in product(alphas, gammas, batch_sizes)]
        params_mat = np.array(params_mat, dtype=theano.config.floatX)
        params_mat = np.column_stack((params_mat,
                                      zerosX(params_mat.shape[0]),
                                      zerosX(params_mat.shape[0]),
                                      zerosX(params_mat.shape[0])))

        for param_idx in xrange(params_mat.shape[0]):
            # load parameters for neural network model
            alpha = params_mat[param_idx, 0]
            gamma = params_mat[param_idx, 1]
            batch_size = int(params_mat[param_idx, 2])
            shape = (batch_size, x_train.shape[1])

<<<<<<< HEAD
            # choose n_hidden nodes according to ...
            n_hidden = int((n_obs / depth) / (alpha*(n_inputs+n_outputs)))

            for i in range(1, depth-1):
                widths[i] = n_hidden

            model_str = ('\nalpha {} gamma {} batch size {} '
                         'n_hidden {} depth {}' 
                         '\nnonlins {}'
                         '\ndrops {}'.format(alpha, gamma, batch_size,
                                             n_hidden, depth, nonlins,
                                             drops))
            print model_str

            # specify input and target theano data types
            input_var = T.fmatrix('input_var')
            # input_var = T.fvector()
            target_var = T.ivector()
            # target_var = T.fmatrix()

            # build neural network model
            network = build_network(input_var, shape, nonlins, depth, widths,
                                    drops)

            # create loss expression for training
            """
            py_x = model(input_var, w_h, w_o)
            y_x = T.argmax(py_x, axis=1)

            cost = T.mean(T.nnet.categorical_crossentropy(py_x, target_var),
                          dtype=theano.config.floatX)
            """
            prediction = lasagne.layers.get_output(network)
            loss = lasagne.objectives.categorical_crossentropy(prediction,
                                                               target_var)
            loss = loss.mean()

            # create paraneter update expressions for training
            """
            params = [w_h, w_o]
            updates = sgd(cost, params, gamma=gamma)
            """
            params = lasagne.layers.get_all_params(network, trainable=True)
            updates = lasagne.updates.adadelta(loss, params,
                                               learning_rate=gamma,
                                               rho=decay_rate)

            # create loss expression for validation and classification accuracy
            # Deterministic forward pass to disable droupout layers
            test_prediction = lasagne.layers.get_output(network,
                                                        deterministic=True)
            test_loss = lasagne.objectives.categorical_crossentropy(
                                                            test_prediction,
                                                            target_var)
            test_loss = test_loss.mean()
            test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1),
                                   target_var), dtype=theano.config.floatX)

            # compile functions for performing training step and returning
            # corresponding training loss
            train_fn = theano.function(inputs=[input_var, target_var],
                                       outputs=loss,
                                       updates=updates,
                                       allow_input_downcast=True)

            # compile a function to compute the validation loss and accuracy
            val_fn = theano.function(inputs=[input_var, target_var],
                                     outputs=[test_loss, test_acc],
                                     allow_input_downcast=True)

            # create kfold iterator
            kf = KFold(x_train.shape[0], n_folds=n_folds)
            error_rates = []
            val_costs = []
            running_time = []

            fold = 1
            start_time = time.time()
            for train_idx, val_idx in kf:
                for i in range(max_epoch):
                    train_err = 0
                    train_batches = 0
<<<<<<< HEAD
                    for start, end in batch_ids(batch_size, x_train, 
=======
                    start_time = time.time()
                    for start, end in batch_ids(batch_size, x_train,
>>>>>>> 2c3917e6d43bd76762f910fe6aa0e0da74ccb099
                                                train_idx):
                        train_err += train_fn(x_train[train_idx][start:end],
                                              y_train[train_idx][start:end])
                        train_batches += 1

                    val_err = 0
                    val_acc = 0
                    val_batches = 0
                    for start, end in batch_ids(batch_size, x_train,
                                                train_idx):
                        err, acc = val_fn(x_train[val_idx], y_train[val_idx])
                        val_err += err
                        val_acc += acc
                        val_batches += 1

<<<<<<< HEAD
                    error_rate = (1 - (val_acc / val_batches)) * 100
                    val_loss = val_err / val_batches

                    print("Final results:")
                    print("  val loss:\t\t\t{:.6f}".format(val_loss))
                    print("  val error rate:\t\t{:.2f} %".format(error_rate))

                error_rates.append(error_rate)
                val_costs.append(val_err)
                running_time.append(np.around((time.time() - 
                                               start_time) / 60., 1))
                fold += 1

            params_mat[param_idx, 3] = np.mean(error_rates)
            params_mat[param_idx, 4] = np.mean(val_costs)
            params_mat[param_idx, 5] = np.mean(running_time)

            print('alpha {} gamma {} batchsize {} error rate {} '
                  'validation cost {} ' 
                  'running time {}'.format(params_mat[param_idx,0],
                                           params_mat[param_idx,1],
                                           params_mat[param_idx,2],
                                           params_mat[param_idx,3],
                                           params_mat[param_idx,4],
                                           params_mat[param_idx,5]))


        # Save params matrix to disk
        params_mat.dump(('results/train/{}'
                         '_results.np').format(train_filename[:-3]))
=======
                    val_err_rate = (1 - (val_acc / val_batches)) * 100
                    val_loss = val_err / val_batches

                    print("Final results:")
                    print("  test loss:\t\t\t{:.6f}".format(val_loss))
                    print("  test error rate:\t\t{:.2f} %".format(val_err_rate))

                error_rates.append(1 - test_acc)
                test_costs.append(val_err)
                running_time.append(np.around((time.time() - start_time) / 60.,
                                              1))
                fold += 1

            params_mat[param_idx, 3] = np.mean(val_err_rate)
            params_mat[param_idx, 4] = np.mean(val_loss)
            params_mat[param_idx, 5] = np.mean(running_time)

            print('alpha {} gamma {} batchsize {}'
                  'error rate {} test cost {}'
                  'running time {}'.format(params_mat[param_idx, 0],
                                           params_mat[param_idx, 1],
                                           params_mat[param_idx, 2],
                                           params_mat[param_idx, 3],
                                           params_mat[param_idx, 4],
                                           params_mat[param_idx, 5]))

        # Save params matrix to disk
        params_mat.dump(('results/train/{}_results.np'
                         ''.format(train_filename[:-3])))
>>>>>>> 2c3917e6d43bd76762f910fe6aa0e0da74ccb099
