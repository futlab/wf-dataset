import numpy as np
import sys
from math import sin, cos, pi
from os import listdir
from os.path import join
from skimage import img_as_float
from skimage.io import imread
from skimage.transform import rotate, rescale
import random
from queue import Queue, Full
from threading import Thread
#from scipy.ndimage.interpolation import rotate, zoom
from pympler import summary, muppy
import _thread

shutdown = False


def shutdown_generator():
    global shutdown
    shutdown = True


def train_generator_queue(out_channels, classes, size, batch_size=None, dump_mem=False, every_flip=False, root=None, workers=1, gain=None):

    def load_classes(folder, class_list):
        if root is not None:
            folder = join(root, folder)
        fns = []
        for c in class_list:
            fns += [join(c, f) for f in listdir(join(folder, c)) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]

        def load(fn):
            img = img_as_float(imread(folder + fn))
            assert img.shape[0] > size[0] and img.shape[1] > size[1]
            return img
        return list(map(load, fns))

    def prep_in(i):
        if len(i.shape) == 2 or i.shape[2] == 3:
            return i
        return i[:, :, :3]

    def prep_out(o):
        if gain is not None:
            o *= gain
            o[o > 1.0] = 1.0

        if len(o.shape) == 2: # единственный канал
            result = np.reshape(o, (o.shape[0], o.shape[1], 1))
            if out_channels == 1:
                return result
            if out_channels == 2:
                return np.concatenate((result, 1.0 - result), axis=2)

        result = o[:, :, :out_channels - 1]
        if o.shape[2] == 4 and out_channels == 4:
            alpha = o[:, :, 3]
            r = np.minimum(o[:, :, 0], alpha)
            g = np.minimum(o[:, :, 1], alpha)
            b = np.minimum(o[:, :, 2], alpha)
            return np.stack((r, g, b, 1.0 - alpha), axis=2)
        alpha = 1.0 - np.sum(result, 2, keepdims=True)
        alpha[alpha < 0.0] = 0.0
        return np.concatenate((result, alpha), axis=2)

    def test_angle(a, i, o): # angle, (height, width) of input, (height, width) of output
        a *= -pi / 180
        cs, sn = cos(a), sin(a)
        acs, asn = abs(cs), abs(sn)
        rh = i[0] - o[0] * acs - o[1] * asn  # moving range
        rw = i[1] - o[1] * acs - o[0] * asn
        return rh >= 0 and rw >= 0

    def gen_pos(a, i, o, count): # angle, (height, width) of input, (height, width) of output
        results = []

        def test_pos(p):
            for r in results:
                if abs(r[0] - p[0]) < o[0] * 0.8 and abs(r[1] - p[1]) < o[1] * 0.8:
                    return False
            return True

        a *= -pi / 180
        cs, sn = cos(a), sin(a)
        acs, asn = abs(cs), abs(sn)
        rh = i[0] - o[0] * acs - o[1] * asn  # moving range
        rw = i[1] - o[1] * acs - o[0] * asn
        ih = i[0] * acs + i[1] * asn  # rotated input size
        iw = i[1] * acs + i[0] * asn
        for c in range(count * 2):
            dh = rh * (random.random() - 0.5)  # select offset
            dw = rw * (random.random() - 0.5)
            rdh = cs * dh - sn * dw  # rotated offset
            rdw = cs * dw + sn * dh
            r = (int((ih - o[0]) / 2 - rdh), int((iw - o[1]) / 2 + rdw))
            if test_pos(r):
                results.append(r)
                yield r
                if len(results) > count:
                    break

    def rotate_f(i, a):
        img = rotate(i, a, resize=True)
        img[img > 1.0] = 1.0
        img[img < 0.0] = 0.0
        return img

    def put(q, v):
        while True:
            try:
                q.put(v, timeout=1)
                return
            except Full:
                if shutdown:
                    sys.exit(0)
                else:
                    continue

    def generate_loop(dump_mem):
        try:
            #raise MemoryError()
            class_tuple = classes if isinstance(classes, tuple) else (classes, classes)
            inputs = list(map(prep_in, load_classes('input/', class_tuple[0])))
            outputs = list(map(prep_out, load_classes('output/', class_tuple[1])))
            random.seed(1)
            area = size[0] * size[1]
            while not shutdown:
                results = []
                metrics = []
                msum = np.ones(out_channels)
                while len(results) < 20:
                    n = random.randint(0, len(inputs) - 1)
                    i = inputs[n]
                    o = outputs[n]
                    scale = 3.0 ** (0.4 - random.random())
                    if scale * i.shape[0] > size[0] and scale * i.shape[1] > size[1] and random.random() > 0.5:
                        i = rescale(i, scale)
                        o = rescale(o, scale)
                    i_shape = i.shape
                    a = -30 + 60.0 * random.random()
                    while not test_angle(a, i_shape, size):
                        a *= 0.5
                    i = rotate_f(i, a)
                    o = rotate_f(o, a)
                    count = int(i_shape[0] * i_shape[1] / area * 3)
                    for p in gen_pos(a, i_shape, size, count):
                        x, y = p[1], p[0]
                        ip = i[y:y + size[0], x:x + size[1], :]
                        op = o[y:y + size[0], x:x + size[1], :]
                        if not every_flip and random.randint(0, 10) > 5:
                            ip = np.flip(ip, 1)
                            op = np.flip(op, 1)
                        m = np.sum(op, axis=(0, 1))
                        if m.sum() == 0:
                            continue
                        msum += m
                        metrics.append((len(results), m))
                        results.append((ip, op))
                metrics = sorted(metrics, key=lambda m: -np.sum((m[1] / msum)[:out_channels - 1]))
                metrics = metrics[: int(len(metrics) * 0.5)] # Reduce number of empty outputs
                random.shuffle(metrics)
                for a in metrics:
                    r = results[a[0]]
                    yield r
                    if every_flip:
                        yield np.flip(r[0], 1), np.flip(r[1], 1)

                if dump_mem:
                    summary.print_(summary.summarize(muppy.get_objects()))
        except MemoryError:
            print('Memory error...')
            _thread.interrupt_main()
            #sys.exit(2)

    def generate_batch_thread(q, dump_mem):
        if batch_size is None:
            for r in generate_loop(dump_mem=dump_mem):
                put(q, r)
        else:
            bi = []
            bo = []
            for i, o in generate_loop(dump_mem=dump_mem):
                bi.append(i)
                bo.append(o)
                if len(bi) >= batch_size:
                    put(q, (np.array(bi), np.array(bo)))
                    bi.clear()
                    bo.clear()

    queue = Queue(maxsize=50 if batch_size is None else int(50 / batch_size))
    for w in range(workers):
        thread = Thread(target=generate_batch_thread, args=(queue, dump_mem))
        thread.start()
    return queue


def train_generator(*args, **kwargs):
    queue = train_generator_queue(*args, **kwargs)
    try:
        while True:
            yield queue.get()
    except MemoryError:
        print('Memory error...')
        _thread.interrupt_main()
        sys.exit(2)


def inverse_channel(image, channel):
    result = []
    if channel > 0:
        result.append(image[:, :, :channel])
    c = image[:, :, channel:channel + 1]
    mx = np.max(c)
    #if mx == 0:
    #    if c.dtype == fl
    result.append(mx - c)
    if image.shape[2] > channel + 1:
        result.append(image[:, :, channel + 1:])
    return np.concatenate(result, 2)


def load_samples(folder, inv_channel=None, set_channels_count=None, aug_inv=None, gain=None):
    result = [img_as_float(imread(join(folder, f))) for f in listdir(folder) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
    for i in range(len(result)):
        shape = result[i].shape
        if len(shape) == 2:  # единственный канал
            result[i] = np.reshape(result[i], (shape[0], shape[1], 1))
        if gain is not None:
            r = result[i] * gain
            r[r > 1.0] = 1.0
            result[i] = r
    if inv_channel is not None:
        result = list(map(lambda i: inverse_channel(i, inv_channel), result))
    if set_channels_count is not None:
        result = list(map(lambda i: i[:, :, :3], result))
    if aug_inv is not None:
        result = list(map(lambda i: np.concatenate((i, 1.0 - i[:, :, aug_inv:1]), axis=2), result))
    return np.array(result)
