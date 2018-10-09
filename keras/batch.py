import numpy as np
import json
from argparse import ArgumentParser
from skimage import img_as_float
from skimage.io import imread, imsave
from os import listdir
from os.path import join, isfile
from matplotlib import pyplot as plt
from keras import models

parser = ArgumentParser(description='Learning neural network')
parser.add_argument('name', type=str, default='boe8_2x2_16-128', help='Name of network')
parser.add_argument('input', type=str, help='Directory to process')
parser.add_argument('output', type=str, help='Directory of output')
args = parser.parse_args()
model_name = args.name
output_dir = args.output

#modelName = 'boe8_2x2_16-128'

models_path = 'models-sam'

with open(join(models_path, model_name + '.json')) as model_file:
    model = models.model_from_json(model_file.read())

state = json.load(open(join(models_path, model_name + '_state.json')))
align = state.get('size', 128)

model.load_weights(join(models_path, model_name + '_best.hdf5'))


def mix(i, r):
    i = np.sum(i, 2) / 3.0


    return i


for fn in listdir(args.input):
    ifn = join(args.input, fn)
    if isfile(ifn) and (fn.endswith('.jpg') or fn.endswith('.png') or fn.endswith('.jpeg')):
        print('Processing: ' + ifn)
        image = img_as_float(np.array([imread(ifn)]))
        image = image[:, :, :, :3]
        size = image.shape
        h = int(size[1] / align) * align
        w = int(size[2] / align) * align
        image = image[:, :h, :w]
        result = model.predict(image)[0, :, :, 0]
        imsave(join(args.output, fn), result)

        #imsave(join(args.output, 'm_' + fn), mix(image, result))


print('Completed')
