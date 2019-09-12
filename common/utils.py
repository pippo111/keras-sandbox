import numpy as np
import os
import nibabel as nib
import random
from scipy.ndimage import zoom, rotate, shift, distance_transform_edt as distance
from keras.layers.convolutional import ZeroPadding3D, Cropping3D
from keras.backend import int_shape

""" Divide background and structure by labels
Returns binarized image as 0|255
"""
def binarize(data: np.ndarray, labels: list) -> np.ndarray:
    for label in labels:
        data[data == label] = 255.0

    data[data != 255.0] = 0.0

    return data.astype(np.uint8)

""" Resizes image to desired shape 2d/3d
"""
def resize(data: np.ndarray, new_shape: tuple) -> np.ndarray:
    old_shape = data.shape
    ratios = tuple(new_dim / old_dim for new_dim, old_dim in zip(new_shape, old_shape))

    resized_data = zoom(data, ratios, order=0)

    return resized_data.astype(np.float32)

""" Slices cuboid into smaller pieces
    Can be undone by 'uncubify' fn
"""
def slice_cuboid(inputs: np.ndarray, new_shape: tuple) -> np.ndarray:
    W, H, D = inputs.shape
    w, h, d = new_shape

    outputs = inputs.reshape(W//w, w, H//h, h, D//d, d)
    outputs = outputs.transpose(0,2,4,1,3,5)

    outputs = outputs.reshape(-1, w, h, d)

    return outputs

"""Normalizes input image to range 0-255
    Can be undone by 'uncubify' fn
"""
def norm_to_uint8(data: np.ndarray) -> np.ndarray:
    max_value = data.max()
    if not max_value == 0:
        data = data / max_value

    data = 255 * data
    img = data.astype(np.uint8)
    return img

def one_hot_encode(seg, C):
    seg = seg.squeeze()
    res = np.stack([seg == c for c in range(C)], axis=-1).astype(np.float32)

    return res

def augment_3d(x, y):
    rotate_range = 5
    shift_range = 0.1

    random_rotate = random.randint(0, rotate_range)
    random_shift = random.uniform(0.0, shift_range)
    shift_px = round(x.shape[0] * random_shift)

    x = rotate(x, random_rotate, order=0, reshape=False)
    y = rotate(y, random_rotate, order=0, reshape=False)

    x = shift(x, (shift_px, 0, 0), order=0)
    y = shift(y, (shift_px, 0, 0), order=0)

    return x, y

def load_image_data(filename: str, path: str = '.') -> np.ndarray:
    full_path = os.path.join(path, filename)
    img = nib.load(full_path)
    img_data = img.get_fdata(dtype=np.float32)

    return img_data

def calc_fit_pad(width, height, depth, n_layers):
    divider = 2 ** n_layers
    w_pad, h_pad, d_pad = 0, 0, 0

    w_rest = width % divider
    h_rest = height % divider
    d_rest = depth % divider

    if w_rest:
        w_pad = (divider - w_rest) // 2
    if h_rest:
        h_pad = (divider - h_rest) // 2
    if d_rest:
        d_pad = (divider - d_rest) // 2

    return w_pad, h_pad, d_pad

def pad_to_fit(inputs, n_layers=4):
    width = int_shape(inputs)[1]
    height = int_shape(inputs)[2]
    depth = int_shape(inputs)[3]

    w_pad, h_pad, d_pad = calc_fit_pad(width, height, depth, n_layers)

    x = ZeroPadding3D((w_pad, h_pad, d_pad))(inputs)
    return x

def crop_to_fit(inputs, outputs, n_layers=4):
    width = int_shape(inputs)[1]
    height = int_shape(inputs)[2]
    depth = int_shape(inputs)[3]

    w_pad, h_pad, d_pad = calc_fit_pad(width, height, depth, n_layers)

    x = Cropping3D((w_pad, h_pad, d_pad))(outputs)
    return x

def calc_confusion_matrix(mask, pred):
    combined = mask * 2 + pred

    fp_total = 0 # false positive total pixels
    fn_total = 0 # false negative total pixels
    tn_total = 0 # true negative total pixels
    tp_total = 0 # true positive total pixels

    for image in combined:
        fp = np.count_nonzero(image == 1.0) # false positive (red)
        fn = np.count_nonzero(image == 2.0) # false negative (yellow)
        tp = np.count_nonzero(image == 3.0) # true positive (green)
        tn = np.count_nonzero(image == 0.0) # true negative (black)

        fp_total += fp
        fn_total += fn
        tn_total += tn
        tp_total += tp

    fpr_perc = fp_total / (fp_total + tn_total) if tn_total != 0 or fp_total != 0 else 0
    fnr_perc = fn_total / (fn_total + tp_total) if tp_total != 0 or fn_total != 0 else 0

    f_total = fp_total + fn_total

    fpr_perc = '{:.0%}'.format(fpr_perc)
    fnr_perc = '{:.0%}'.format(fnr_perc)

    return fpr_perc, fnr_perc, fp_total, fn_total, f_total

def calc_precision(mask, pred):
    combined = mask * 2 + pred

    fp_total = 0 # false positive total pixels
    tp_total = 0 # true positive total pixels

    for image in combined:
        fp = np.count_nonzero(image == 1.0) # false positive (red)
        tp = np.count_nonzero(image == 3.0) # true positive (green)

        fp_total += fp
        tp_total += tp

    precision = tp_total / (tp_total + fp_total)

    return precision

def calc_recall(mask, pred):
    combined = mask * 2 + pred

    fn_total = 0 # false negative total pixels
    tp_total = 0 # true positive total pixels

    for image in combined:
        fn = np.count_nonzero(image == 2.0) # false negative (yellow)
        tp = np.count_nonzero(image == 3.0) # true positive (green)

        fn_total += fn
        tp_total += tp

    recall = tp_total / (tp_total + fn_total)

    return recall

def calc_f1score(precision, recall):
    f1 = 2 * ( (precision*recall) / (precision+recall) )

    return f1

def get_all_gen_items(generator):
    X_items = list()
    y_items = list()

    for X, y in generator:
        for item in X:
            X_items.append(item)

        for item in y:
            y_items.append(item)

    return np.array(X_items), np.array(y_items)

def pad_3d(data, w, h, d, cube_dim):
    pad_w = (cube_dim - w) // 2
    pad_h = (cube_dim - h) // 2
    pad_d = (cube_dim - d) // 2

    data = np.pad(
        data,
        [(pad_w, pad_w), (pad_h, pad_h), (pad_d, pad_d)],
        mode='constant',
        constant_values=0
    )
    
    return data



def uncubify(arr, oldshape):
    dummy_N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)    
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)

def calc_weights_generator(generator):
    class_counter = { 'background': 0, 'structure': 0 }

    for dummy_X, y in generator:
        y = np.array(y).argmax(axis=-1)
        nonzeros = np.count_nonzero(y)
        class_counter['background'] += y.size - nonzeros
        class_counter['structure'] += nonzeros
    class_share = { key: value / sum(class_counter.values()) for key, value in class_counter.items() }

    class_weights = { 'background': class_share['structure'], 'structure': class_share['background'] }

    return class_weights