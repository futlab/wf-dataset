import numpy as np
from skimage.io import imsave
from argparse import ArgumentParser
from generator import train_generator, inverse_channel, shutdown_generator
from os import mkdir
from os.path import isdir, isfile, abspath, join

def mkd(f):
    if not isdir(f):
        mkdir(f)


parser = ArgumentParser(description='Learning neural network')
parser.add_argument('name', type=str, default='validation', help='Name of directory')
parser.add_argument('count', type=int, default=100, help='Count of samples')
parser.add_argument('-s', type=int, default=64, help='Height and width of results')
parser.add_argument('-i', type=str, default='input', help='Input class list')
parser.add_argument('-o', type=str, default='output', help='Output class list')
args = parser.parse_args()
folder = abspath(args.name)
mkd(folder)
input_folder = join(folder, 'input')
output_folder = join(folder, 'output')
mkd(input_folder)
mkd(output_folder)
input_classes = args.i.split(',')
output_classes = args.o.split(',')
n = 1
for i, o in train_generator(1, (input_classes, output_classes), (args.s, args.s), root='../'):
    while True:
        name = "%04d.png" % (n,)
        if isfile(join(input_folder, name)):
            n += 1
        else:
            break
    if n > args.count:
        break
    imsave(join(input_folder, name), i)
    imsave(join(output_folder, name), o[:, :, 0])
    n += 1
print('Terminating...')
shutdown_generator()
