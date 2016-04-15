""" Adaptep from Colin Raffel's git repo https://github.com/craffel/"""
import numpy as np
import theano
from theano import tensor as T
import lasagne
import nnet_utils


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


def train(data, layers, updates_fn, batch_size=16, epoch_size=128,
          initial_patience=1000, improvement_threshold=0.99,
          patience_increase=5, max_iter=100000):

    # specify input and target theano data types
    input_var = T.fmatrix()
    target_var = T.ivector()

    # create a cost expression for training
    prediction = lasagne.layers.get_output(layers, input_var)
    cost = lasagne.objectives.categorical_crossentropy(
        prediction, target_var)
    cost = cost.mean()

    # create parameter update expressions for training
    params = lasagne.layers.get_all_params(layers, trainable=True)
    updates = updates_fn(cost, params)

    # compile functions for performing training step and returning
    # corresponding training cost
    train_fn = theano.function(inputs=[input_var, target_var],
                               outputs=cost,
                               updates=updates,
                               allow_input_downcast=True,
                               on_unused_input='warn')

    # create cost expression for validation
    # deterministic forward pass to disable droupout layers
    val_prediction = lasagne.layers.get_output(layers, input_var,
                                               deterministic=True)
    val_cost = lasagne.objectives.categorical_crossentropy(
        val_prediction, target_var)
    val_cost = val_cost.mean()
    val_obj_fn = T.mean(T.neq(T.argmax(val_prediction, axis=1),
                              target_var), dtype=theano.config.floatX)

    # compile a function to compute the validation cost and objective function
    validate_fn = theano.function(inputs=[input_var, target_var],
                                  outputs=[val_cost, val_obj_fn],
                                  allow_input_downcast=True)

    # create data iterators
    train_data_iterator = nnet_utils.get_next_batch(
        data['train'][:, :-1], data['train'][:, -1], batch_size, max_iter)

    patience = initial_patience
    current_val_cost = np.inf
    train_cost = 0.0

    for n, (x_batch, y_batch) in enumerate(train_data_iterator):
        train_cost += train_fn(x_batch, y_batch)

        # Stop training if NaN is encountered
        if not np.isfinite(train_cost):
            print 'Bad training er {} at iteration {}'.format(train_cost, n)
            break

        if n and not (n % epoch_size):
            epoch_result = {'iteration': n,
                            'train_cost': train_cost / float(epoch_size),
                            'validate_cost': 0.0,
                            'validate_objective': 0.0}

            # compute validation cost and objective
            cost, obj = validate_fn(data['validate'][:, :-1],
                                    data['validate'][:, -1])

            epoch_result['validate_cost'] = float(cost)
            epoch_result['validate_objective'] = float(obj)

            # Test whether this validate cost is the new smallest
            if epoch_result['validate_cost'] < current_val_cost:
                # To update patience, we must be smaller than
                # improvement_threshold*(previous lowest validation cost)
                patience_cost = improvement_threshold*current_val_cost
                if epoch_result['validate_cost'] < patience_cost:
                    # Increase patience by the supplied about
                    patience += epoch_size*patience_increase
                # Even if we didn't increase patience, update lowest valid cost
                current_val_cost = epoch_result['validate_cost']
            # Store patience after this epoch
            epoch_result['patience'] = patience

            if n > patience:
                break

            yield epoch_result


def build_general_network(input_shape, n_layers, widths,
                          non_linearities, drop_out=True):
    """
    Parameters
    ----------
    input_shape : tuple of int or None (batchsize, rows, cols)
        Shape of the input. Any element can be set to None to indicate that
        dimension is not fixed at compile time

    """

    # GlorotUniform is the default mechanism for initializing weights
    for i in range(n_layers):
        if i == 0:  # input layer
            layers = lasagne.layers.InputLayer(shape=input_shape)
        else:  # hidden and output layers
            layers = lasagne.layers.DenseLayer(layers,
                                               num_units=widths[i],
                                               nonlinearity=non_linearities[i])
            if drop_out and i < n_layers-1:  # output layer has no dropout
                layers = lasagne.layers.DropoutLayer(layers, p=0.5)
    return layers
