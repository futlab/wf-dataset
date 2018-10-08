import random
import numpy as np
from os.path import join
from keras.models import Sequential, model_from_json
from keras.layers import Input, Dropout, Conv2D


def random_model():
    layer_count = random.randint(2, 5)
    layers = [
        #Input(shape=(None, None, 3)),
        Dropout(0.1, input_shape=(None, None, 3))
    ]
    for i in range(layer_count):
        name = 'conv' + str(i + 1)
        size = (1 + 2 * random.randint(0, 2), 1 + 2 * random.randint(0, 2))
        layers.append(Conv2D(random.randint(1, 4), size, padding='same', activation='relu', name=name))
    layers.append(Conv2D(1, (1, 1), padding='same', name='decisive', activation='sigmoid'))
    return Sequential(layers)


def model_to_name(model):
    def layer_to_name(layer):
        layer_type = layer.__class__.__name__
        if layer_type == 'Conv2D':
            ks = layer.kernel_size
            return '%X%X%X' % (layer.filters, int((ks[0] - 1) / 2), int((ks[1] - 1) / 2))
        return '?'
    return '-'.join(map(layer_to_name, model.layers[1:-1]))


def load_model(models_folder, model_name):
    try:
        with open(join(models_folder, model_name + '.json')) as model_file:
            model = model_from_json(model_file.read())
    except Exception as e:
        print('Unable to load %s: ' % model_to_name, str(e))
        return None
    try:
        model.load_weights(join(models_folder, model_name + '_best.hdf5'))
    except Exception as e:
        print('Unable to load weights %s: ' % model_to_name, str(e))
    return model


def analyse_model(model):
    params = 0
    for layer in model.layers[1:]:
        layer_type = layer.__class__.__name__
        if layer_type == 'Conv2D':
            params += int(np.prod(layer.weights[0].shape) + np.prod(layer.weights[1].shape))
    return {'param_count': params}


def resize(w, d, axis=0, noise=0.03):
    if d > 0:
        shape = list(w.shape)
        shape[axis] = d
        zeros = np.random.standard_normal(size=shape) * noise
        return np.concatenate((zeros, w, zeros), axis=axis)
    else:
        if axis == 0:
            return w[-d:d, :, :, :]
        if axis == 1:
            return w[:, -d:d, :, :]
        if axis == 2:
            return w[:, :, -d:d, :]
        if axis == 3:
            return w[:, :, :, -d:d]


def unique_name(model, prefix):
    idx = 1
    try:
        while model.get_layer(prefix + str(idx)) is not None:
            idx += 1
    except ValueError:
        return prefix + str(idx)


def change_layer_inputs(model, layer_idx, d, filter_to_remove=-1):
    layer = model.layers[layer_idx]
    layer_type = layer.__class__.__name__
    if layer_type == 'Conv2D':
        w = layer.get_weights()
        shape = list(w[0].shape)
        inputs = shape[2] + d
        print('Mutating layer %d: %srement inputs to %d' % (layer_idx, 'inc' if d > 0 else 'dec', inputs))
        if d > 0:
            shape[2] = d
            w[0] = np.concatenate((w[0], np.random.standard_normal(size=shape) * 0.03), axis=2)
        else:
            w[0] = np.delete(w[0], filter_to_remove, axis=2)
        layer = Conv2D(layer.filters, layer.kernel_size, padding='same', activation='relu',
                                         weights=w)
        shape[2] = inputs
        #layer.build(shape)
    else:
        print('Unable to change_layer_input: unknown class %s', layer_type)
    return layer


def mutate_layer(model, layer_idx):
    layer = model.layers[layer_idx]
    layer_type = layer.__class__.__name__
    next_layer = None
    if layer_type == 'Conv2D':
        w = layer.get_weights()
        kernel = list(layer.kernel_size)
        filters = layer.filters
        target = random.randint(0, 2)
        if target == 0 or target == 1:
            name = 'height' if target == 0 else 'width'
            if random.randint(0, 1) == 1 or kernel[target] == 1:
                print('Mutating layer %d: increment %s to %d' % (layer_idx, name, kernel[target] + 2))
                w[0] = resize(w[0], 1, axis=target)
                kernel[target] += 2
            else:
                print('Mutating layer %d: decrement %s to %d' % (layer_idx, name, kernel[target] - 2))
                w[0] = resize(w[0], -1, axis=target)
                kernel[target] -= 2
        elif target == 2:
            if random.randint(0, 1) == 1 or filters == 1:
                filters += 1
                print('Mutating layer %d: increment filters to %d' % (layer_idx, filters))
                shape = list(w[0].shape)
                shape[3] = 1
                w[0] = np.concatenate((w[0], np.random.standard_normal(size=shape) * 0.03), axis=3)
                w[1] = np.concatenate((w[1], np.random.rand(1) * 0.03), axis=0)
                next_layer = change_layer_inputs(model, layer_idx + 1, 1)
            else:
                filters -= 1
                fr = random.randint(0, filters)
                print('Mutating layer %d: decrement filters to %d' % (layer_idx, filters))
                w[0] = np.delete(w[0], fr, axis=3)
                w[1] = np.delete(w[1], fr)
                next_layer = change_layer_inputs(model, layer_idx + 1, -1, filter_to_remove=fr)
        layer = Conv2D(filters, kernel, padding='same', activation='relu',
                                         weights=w)
        #layer.build(model.layers[layer_idx - 1].output_shape)
        return layer, next_layer


def replace_layer(model, layer_idx, new_layers):
    layers = [l for l in model.layers]
    layer, next_layer = new_layers
    for i in range(1, len(layers)):
        layers[i] = layer if i == layer_idx else layers[i]
        layers[i] = next_layer if next_layer is not None and i == layer_idx + 1 else layers[i]
        if i >= 1:
            layers[i].name = 'conv2d_' + str(i)
    return Sequential(layers)


def insert_layer(model, layer_idx):
    print('Inserting layer after %d' % layer_idx)
    input_shape = model.layers[layer_idx].output_shape
    filters = input_shape[3]
    w = [
        np.reshape(np.identity(filters), (1, 1, filters, filters))
        + np.random.randn(1, 1, filters, filters) * 0.05,
        np.random.randn(filters) * 0.03
    ]
    layer = Conv2D(filters, 1, padding='same', activation='relu', weights=w, name=unique_name(model, 'conv2d_'))
    layers = [l for l in model.layers]
    layers.insert(layer_idx + 1, layer)
    return Sequential(layers)


def mutate(models_folder, model_name):
    model = load_model(models_folder, model_name)
    if model is None:
        return None
    action = random.random()
    layer_idx = random.randint(1, len(model.layers) - 2)
    if action > 0.8:
        return insert_layer(model, layer_idx)
    else:
        return replace_layer(model, layer_idx, mutate_layer(model, layer_idx))
