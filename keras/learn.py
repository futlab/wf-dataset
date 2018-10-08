import sys
import random
from os import listdir
from os.path import join
from time import sleep, time
from keras import models
from keras.layers import Input, Conv2D, Activation, concatenate, Reshape, Permute, MaxPooling2D, UpSampling2D, BatchNormalization, GlobalMaxPooling2D, Dropout, ZeroPadding2D
from keras.optimizers import SGD, Adagrad
import json
from mutator import mutate, random_model, model_to_name, analyse_model
from keras.utils import plot_model
from generator import train_generator_queue, shutdown_generator, load_samples

models_dir = 'models-sam'
sample_x = load_samples('validation-sam/input', set_channels_count=3)
sample_y = load_samples('validation-sam/output', gain=2.0)

validate = True
classes = ['sam']
max_epochs = 200
train_queue = train_generator_queue(1, (classes, classes), (32, 32), 40, every_flip=True, root='..', gain=2.0)
train_data = None

root_code = [2, 3, 1,
             4, 1, 15,
             8, 3, 3]


def load_models(folder):
    model_states = {}
    for f in listdir(folder):
        if f.endswith('_state.json'):
            try:
                s = json.load(open(join(folder, f)))
                model_states.update({f[:-11] : s})
            except json.JSONDecodeError:
                print('Unable to read model state: ' + f)
    return model_states


def get_best_models(ms, trim=10):
    def bl(m):
        return ms[m]['best_loss']
    bm = sorted(list(ms.keys()), key=bl)
    if trim is not None and len(bm) > trim:
        bm = bm[:trim]
    return bm


def choose_parent_model(models, names):
    metrics = [1000.0 / models[n].get('param_count', 1000) for n in names]
    r = random.random() * sum(metrics)
    t = 0
    for i in range(len(names)):
        t += metrics[i]
        if t >= r:
            return names[i]


model_states = load_models(models_dir)
best_models = get_best_models(model_states)


def train(model, queue, models_folder='models', parent=None, epochs=20):
    model_name = model_to_name(model)
    state = model_states.get(model_name, None)
    if state is not None:
        epoch = state.get('epoch', 0)
        best_loss = state.get('best_loss', 1E6)
        if epoch >= max_epochs:
            print('Model %s: skipping' % model_name)
            return best_loss
        print('Model %s: loading' % model_name)
        with open(join(models_folder, model_name + '.json')) as model_file:
            model = models.model_from_json(model_file.read())
        try:
            model.load_weights(join(models_folder, model_name + '.hdf5'))
            print('Weights loaded')
        except OSError:
            pass
    else:
        print('Model %s: new' % model_name)
        model.summary()
        state = {'epoch': 0, 'parent': parent}
        state.update(analyse_model(model))
        epoch = 0
        best_loss = 1E6
        with open(join(models_folder, model_name + '.json'), 'w') as outfile:
            outfile.write(json.dumps(json.loads(model.to_json()), indent=2))

    optimizer = Adagrad(lr=2E-3) # SGD(lr=0.0001, momentum=0.95, decay=0.0005, nesterov=False)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    def train_on_queue(queue, begin_epoch, end_epoch, best_loss, updates=100):
        epoch = begin_epoch
        global train_data
        u, g = 0, 0
        loss, acc = 0, 0
        t = None
        while True:
            if u >= updates:
                loss /= u
                u = 0
                if validate:
                    validation_loss, validation_acc = model.test_on_batch(sample_x, sample_y)
                    print('%s - epoch %d completed. vl: %.3f, va: %.3f, tl: %.3f, gets: %d, time: %.1fs'
                          % (model_name, epoch, validation_loss, validation_acc, loss, g, time() - t))
                    t = None
                    validation_loss = float(validation_loss)
                    if validation_loss < best_loss:
                        print('Best loss!')
                        best_loss = validation_loss
                        model.save_weights(join(models_folder, model_name + '_best.hdf5'))
                else:
                    print('%s - epoch %d completed. tl: %.3f, gets: %d, time: %.1fs'
                          % (model_name, epoch, loss, g, time() - t))
                g = 0
                epoch += 1
                if epoch >= end_epoch:
                    return best_loss
            if not queue.empty():
                train_data = queue.get()
                g += 1
            if train_data is None:
                print('Waiting for data...')
                sleep(5)
            else:
                if t is None:
                    t = time()
                (l, a) = model.train_on_batch(train_data[0], train_data[1])
                loss += l
                acc += a
                u += 1

    while state['epoch'] < epochs:
        best_loss = train_on_queue(train_queue, epoch, epoch + 5, best_loss)
        epoch += 5
        state.update({'epoch': epoch, 'best_loss': best_loss})
        model.save_weights(join(models_folder, model_name + '.hdf5'))
        json_data = json.dumps(state, indent=2)
        with open(join(models_folder, model_name + '_state.json'), 'w') as outfile:
            outfile.write(json_data)
    model_states.update({model_name : state})
    return best_loss

try:
    parent_name = None
    while True:
        if bool(model_states) and random.random() > 0.1:
            best_models = get_best_models(model_states)
            parent_name = choose_parent_model(model_states, best_models)
            print('Model %s: mutate' % parent_name)
            current_model = mutate(models_dir, parent_name)
            if current_model is None:
                continue
        else:
            print('Random model')
            parent_name = 'random'
            current_model = random_model()
        train(current_model, train_queue, parent=parent_name, epochs=200, models_folder=models_dir)
    shutdown_generator()
except Exception as e: # KeyboardInterrupt or MemoryError or OpError:
    shutdown_generator()
    print('Exception: ', str(e))
    print('Terminating...')
    sys.exit(1)
