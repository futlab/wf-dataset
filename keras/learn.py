import sys
from os import listdir
from os.path import join
from time import sleep, time
from keras import models
from keras.layers import Input, Conv2D, Activation, concatenate, Reshape, Permute, MaxPooling2D, UpSampling2D, BatchNormalization, GlobalMaxPooling2D, Dropout, ZeroPadding2D
from keras.optimizers import SGD, Adagrad
import json
from keras.utils import plot_model
from generator import train_generator_queue, shutdown_generator, load_samples

sample_x = load_samples('validation/input', set_channels_count=3)
sample_y = load_samples('validation/output')

validate=True
train_queue = train_generator_queue(1, (['sam'], ['sam']), (64, 64), 10, every_flip=True, root='..')
train_data = None

root_code = [4, 3, 1,
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


model_states = load_models('models')


def build(code):
    inputs = Input(shape=(None, None, 3))
    data = Dropout(0.1)(inputs)
    data = Conv2D(code[0], (code[1], code[2]), padding='same', activation='relu', name='prefilter')(data)
    code = code[3:]
    data = Conv2D(code[0], (code[1], code[2]), padding='same', activation='relu', name='symm_find')(data)
    code = code[3:]
    data = Conv2D(code[0], (code[1], code[2]), padding='same', activation='relu', name='seq_find')(data)
    outputs = Conv2D(1, (1, 1), padding='same', name='decisive', activation='sigmoid')(data)
    return models.Model(inputs=inputs, outputs=outputs)

def code_to_name(code):
    name = ''
    while len(code) > 0:
        if name != '':
            name = name + '-'
        name = name + '%X%X%X' % tuple(code[:3])
        code = code[3:]
    return name;

def train(code, queue, epochs=20):
    model_name = code_to_name(code)
    state = {'epoch': 0, 'size': 128}
    epoch = 0
    best_loss = 1E6
    # TODO: verify model existance
    model = build(code)
    model.summary()
    with open('models/' + model_name + '.json', 'w') as outfile:
        outfile.write(json.dumps(json.loads(model.to_json()), indent=2))
    optimizer = Adagrad(lr=1E-3) # SGD(lr=0.0001, momentum=0.95, decay=0.0005, nesterov=False)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    def train_on_queue(queue, begin_epoch, end_epoch, best_loss, updates=50):
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
                    print('Epoch %d completed. vl: %.3f, va: %.3f, tl: %.3f, gets: %d, time: %.1fs'
                          % (epoch, validation_loss, validation_acc, loss, g, time() - t))
                    t = None
                    validation_loss = float(validation_loss)
                    if validation_loss < best_loss:
                        print('Best loss!')
                        best_loss = validation_loss
                        model.save_weights('models/' + model_name + '_best.hdf5')
                else:
                    print('Epoch %d completed. tl: %.3f, gets: %d, time: %.1fs'
                          % (epoch, loss, g, time() - t))
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
        #h = train(1)
        state.update({'epoch': epoch, 'best_loss': best_loss})
        model.save_weights('models/' + model_name + '.hdf5')
        json_data = json.dumps(state, indent=2)
        with open('models/' + model_name + '_state.json', 'w') as outfile:
            outfile.write(json_data)


try:
    train(root_code, train_queue, epochs=200)
    shutdown_generator()
except Exception as e: # KeyboardInterrupt or MemoryError or OpError:
    shutdown_generator()
    print('Exception: ', str(e))
    print('Terminating...')
    sys.exit(1)
