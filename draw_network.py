#!/usr/bin/python

"""
From Eben Olson's https://gist.github.com/ebenolson/1682625dc9823e27d771
Functions to create network diagrams from a list of Layers.

Examples:

    Draw a minimal diagram to a pdf file:
        layers = lasagne.layers.get_all_layers(output_layer)
        draw_to_file(layers, 'network.pdf', output_shape=False)

    Draw a verbose diagram in an IPython notebook:
        from IPython.display import Image #needed to render in notebook

        layers = lasagne.layers.get_all_layers(output_layer)
        dot = get_pydot_graph(layers, verbose=True)
        return Image(dot.create_png())
"""

import os
import argparse
import numpy as np
import lasagne
import deepdish
import pydot
import neural_networks
from params import feats_train_folder, MODEL_DIRECTORY, IMAGES_DIRECTORY
from params import nnet_params


def get_hex_color(layer_type):
    """
    Determines the hex color for a layer. Some classes are given
    default values, all others are calculated pseudorandomly
    from their name.
    :parameters:
        - layer_type : string
            Class name of the layer

    :returns:
        - color : string containing a hex color.

    :usage:
        >>> color = get_hex_color('MaxPool2DDNN')
        '#9D9DD2'
    """

    if 'Input' in layer_type:
        return '#A2CECE'
    if 'Conv' in layer_type:
        return '#7C9ABB'
    if 'Dense' in layer_type:
        return '#6CCF8D'
    if 'Pool' in layer_type:
        return '#9D9DD2'
    else:
        return '#{0:x}'.format(hash(layer_type) % 2**24)


def get_pydot_graph(layers, output_shape=True, verbose=False):
    """
    Creates a PyDot graph of the network defined by the given layers.
    :parameters:
        - layers : list
            List of the layers, as obtained from lasange.layers.get_all_layers
        - output_shape: (default `True`)
            If `True`, the output shape of each layer will be displayed.
        - verbose: (default `False`)
            If `True`, layer attributes like filter shape, stride, etc.
            will be displayed.
        - verbose:
    :returns:
        - pydot_graph : PyDot object containing the graph

    """
    pydot_graph = pydot.Dot('Network', graph_type='digraph')
    pydot_nodes = {}
    pydot_edges = []

    for i, layer in enumerate(layers):
        layer_type = '{0}'.format(layer.__class__.__name__)
        key = repr(layer)
        label = layer_type
        color = get_hex_color(layer_type)
        if verbose:
            for attr in ['num_filters', 'num_units', 'ds',
                         'filter_shape', 'stride', 'strides', 'p']:
                if hasattr(layer, attr):
                    label += '\n' + \
                        '{0}: {1}'.format(attr, getattr(layer, attr))
            if hasattr(layer, 'nonlinearity'):
                try:
                    nonlinearity = layer.nonlinearity.__name__
                except AttributeError:
                    nonlinearity = layer.nonlinearity.__class__.__name__
                label += '\n' + 'nonlinearity: {0}'.format(nonlinearity)

        if output_shape:
            label += '\n' + \
                'Output shape: {0}'.format(
                    lasagne.layers.get_output_shape(layer))
        pydot_nodes[key] = pydot.Node(key,
                                      label=label,
                                      shape='record',
                                      fillcolor=color,
                                      style='filled',
                                      )

        if hasattr(layer, 'input_layers'):
            for input_layer in layer.input_layers:
                pydot_edges.append([repr(input_layer), key])

        if hasattr(layer, 'input_layer'):
            pydot_edges.append([repr(layer.input_layer), key])

    for node in pydot_nodes.values():
        pydot_graph.add_node(node)
    for edge in pydot_edges:
        pydot_graph.add_edge(
            pydot.Edge(pydot_nodes[edge[0]], pydot_nodes[edge[1]]))
    return pydot_graph


def draw_to_file(layers, filename, **kwargs):
    """
    Draws a network diagram to a file
    :parameters:
        - layers : list
            List of the layers, as obtained from lasange.layers.get_all_layers
        - filename: string
            The filename to save output to.
        - **kwargs: see docstring of get_pydot_graph for other options
    """
    dot = get_pydot_graph(layers, **kwargs)

    ext = filename[filename.rfind('.') + 1:]
    with open(filename, 'w') as fid:
        fid.write(dot.create(format=ext))


def draw_to_notebook(layers, **kwargs):
    """
    Draws a network diagram in an IPython notebook
    :parameters:
        - layers : list
            List of the layers, as obtained from lasange.layers.get_all_layers
        - **kwargs: see docstring of get_pydot_graph for other options
    """
    from IPython.display import Image  # needed to render in notebook

    dot = get_pydot_graph(layers, **kwargs)
    return Image(dot.create_png())


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
            print("Loading network {}").format(model_name)

            # get shape from train set
            data_shape = np.load(
                os.path.join(feats_train_folder, train_path)).shape
            network = neural_networks.build_general_network(
                (nnet_params['batch_size'], data_shape[1]-1),  # last is target
                nnet_params['n_layers'],
                nnet_params['widths'],
                nnet_params['non_linearities'],
                drop_out=False)

            # load best network model so far
            parameters = deepdish.io.load(
                os.path.join(MODEL_DIRECTORY, model_name+'.h5'))
            lasagne.layers.set_all_param_values(network, parameters)
            # plot model
            print ("Plotting network {}".format(model_name))
            draw_to_file(lasagne.layers.get_all_layers(network),
                         os.path.join(IMAGES_DIRECTORY,
                                      model_name+'_model.png'))
