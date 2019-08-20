import numpy as np
import os
import nibabel as nib
import random
from scipy.ndimage import zoom, rotate, shift
from keras.layers.convolutional import ZeroPadding3D, Cropping3D
from keras.backend import int_shape

def norm_to_uint8(data):
    max_value = data.max()
    if not max_value == 0:
        data = data / max_value

    data = 255 * data
    img = data.astype(np.uint8)
    return img

def convert_to_binary_3d(data, labels):
    binary_data = np.array(
        [[[255.0 if pixel in labels else 0.0 for pixel in row] for row in matrix] for matrix in data]
    ).astype(np.float32)

    return binary_data

def resize_3d(data, width, height, depth):
    in_width, in_height, in_depth = data.shape
    width_ratio = width / in_width
    height_ratio = height /in_height
    depth_ratio = depth / in_depth

    resized_data = zoom(data, (width_ratio, height_ratio, depth_ratio), order=0)

    return resized_data

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

def load_image_data(filename, path='.'):
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

    fpr_array = np.array([]) # false positive ratio (reds)
    fpr_total = np.array([]).astype('int') # false positive total pixels
    fnr_array = np.array([]) # false negative ratio (yellows)
    fnr_total = np.array([]).astype('int') # false negative total pixels

    for image in combined:
        fp = np.count_nonzero(image == 1.0) # false positive (red)
        fn = np.count_nonzero(image == 2.0) # false negative (yellow)
        tp = np.count_nonzero(image == 3.0) # true positive (green)
        tn = np.count_nonzero(image == 0.0) # true negative (black)
        
        fpr = fp / (fp + tn) if tn != 0 or fp != 0 else 0
        fnr = fn / (fn + tp) if tp != 0 or fn != 0 else 0

        fpr_array = np.append(fpr_array, fpr)
        fpr_total = np.append(fpr_total, fp)
        fnr_array = np.append(fnr_array, fnr)
        fnr_total = np.append(fnr_total, fn)

    fpr_perc = '{:.0%}'.format(fpr_array.mean())
    fpr_sum = fpr_total.sum()
    fnr_perc = '{:.0%}'.format(fnr_array.mean())
    fnr_sum = fnr_total.sum()

    return fpr_perc, fnr_perc, fpr_sum, fnr_sum

def get_all_gen_items(generator):
    X_items = list()
    y_items = list()

    for X, y in generator:
        for item in X:
            X_items.append(item)

        for item in y:
            y_items.append(item)

    return np.array(X_items), np.array(y_items)
