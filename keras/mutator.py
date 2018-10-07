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
    with open(join(models_folder, model_name + '.json')) as model_file:
        model = model_from_json(model_file.read())
    model.load_weights(join(models_folder, model_name + '.hdf5'))
    return model


def analyse_model(model):
    params = 0
    for layer in model.layers[1:-1]:
        layer_type = layer.__class__.__name__
        if layer_type == 'Conv2D':
            params += int(np.prod(layer.weights[0].shape) + np.prod(layer.weights[1].shape))
    return {'param_count': params}


def resize(w, d, axis=0):
    if d > 0:
        shape = list(w.shape)
        shape[axis] = d
        zeros = np.zeros(shape)
        return np.concatenate((zeros, w, zeros), axis=axis)
    else:
        if axis == 0:
            return w[d:-d, :, :, :]
        if axis == 1:
            return w[:, d:-d, :, :]
        if axis == 2:
            return w[:, :, d:-d, :]
        if axis == 3:
            return w[:, :, :, d:-d]


def mutate_layer(model, layer_idx):
    layer = model.layers[layer_idx]
    layer_type = layer.__class__.__name__
    if layer_type == 'Conv2D':
        w = layer.get_weights()
        kernel = list(layer.kernel_size)
        target = random.randint(0, 1)
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
            model.layers[layer_idx] = Conv2D(layer.filters, kernel, padding='same', activation='relu', weights=w)
        model.layers[layer_idx].build(model.layers[layer_idx - 1].output_shape)


def insert_layer(model, layer_idx):
    print('Inserting layer after %d' % layer_idx)
    input_shape = model.layers[layer_idx].output_shape
    filters = input_shape[3]
    w = [
        np.reshape(np.identity(filters), (1, 1, filters, filters))
        + np.random.randn(1, 1, filters, filters) * 0.05,
        np.random.randn(filters) * 0.03
    ]
    layer = Conv2D(filters, 1, padding='same', activation='relu', weights=w)
    model.layers.insert(layer_idx + 1, layer)
    layer.build(input_shape)


def mutate(models_folder, model_name):
    model = load_model(models_folder, model_name)
    action = random.random()
    layer_idx = random.randint(1, len(model.layers) - 2)
    if action > 0.9:
        insert_layer(model, layer_idx)
    else:
        mutate_layer(model, layer_idx)
    #elif action < 1.0:#0.85:
    return model
