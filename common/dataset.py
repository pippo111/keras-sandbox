import os
import shutil
import numpy as np
import glob
from PIL import Image
from sklearn.model_selection import train_test_split

from common import data_sequence
from common import utils

class MyDataset():
    def __init__(
            self,
            scans = [],
            labels = [255.0],
            collection_name = 'mindboggle',
            input_label_niftii = 'aseg-in-t1weighted_2std.nii.gz',
            input_image_niftii = 't1weighted_2std.nii.gz',
            scan_shape = (192, 256, 256),
            input_shape = (48, 64, 64),
            batch_size = 32,
            limit = None,
            only_masks = True
        ):
        self.scans = scans
        self.labels = labels
        self.collection_name = collection_name
        self.input_label_niftii = input_label_niftii
        self.input_image_niftii = input_image_niftii
        self.batch_size = batch_size

        self.scan_shape = scan_shape
        self.input_shape = input_shape

        self.limit = limit
        self.only_masks = only_masks

        self.in_dataset_dir = './input/niftii'
        self.out_dataset_dir = os.path.join('./input/datasets', collection_name)

        self.dims = '2d' if 1 in input_shape else '3d'

    """ Public API
    Returns data generators for 3d models
    """
    def get_generators(self):
        X_train, X_valid, X_test, y_train, y_valid, y_test = self.get_train_valid_test_files()

        if self.dims == '2d':
            train_generator = data_sequence.DataSequence2d(X_train, y_train, self.batch_size, augmentation=True)
            valid_generator = data_sequence.DataSequence2d(X_valid, y_valid, self.batch_size, shuffle=False, augmentation=False)
            test_generator = data_sequence.DataSequence2d(X_test, y_test, self.batch_size, shuffle=False, augmentation=False)
        else:
            train_generator = data_sequence.DataSequence3d(X_train, y_train, self.batch_size, augmentation=False)
            valid_generator = data_sequence.DataSequence3d(X_valid, y_valid, self.batch_size, shuffle=False, augmentation=False)
            test_generator = data_sequence.DataSequence3d(X_test, y_test, self.batch_size, shuffle=False, augmentation=False)

        self.train_generator = train_generator
        self.valid_generator = valid_generator
        self.test_generator = test_generator

        return train_generator, valid_generator, test_generator

    """ Public API
        Creates splitted dataset for 2d/3d models
        Ouputs are saved numpy arrays if 3d or png image files if 2d
        2d / 3d is obtained by desired input size
        e.g. (48, 64, 64) will be 3d npy files
        (1, 256, 256) will be 2d png files by axis=0
    """
    def create_dataset(self):
        if os.path.exists(self.out_dataset_dir):
            shutil.rmtree(self.out_dataset_dir)
        os.makedirs(self.out_dataset_dir)

        to_slice = self.scan_shape != self.input_shape

        for scan_name in self.scans:
            full_path = os.path.join(self.in_dataset_dir, scan_name)
            X_data, y_data = self.prepare_images_labels(scan_name, full_path)

            # We are able to chose if scan image should be sliced
            if to_slice:
                sliced_images = utils.slice_cuboid(X_data, self.input_shape)
                sliced_labels = utils.slice_cuboid(y_data, self.input_shape)

                for i, labels in enumerate(sliced_labels):
                    self.save_dataset(sliced_images[i], labels, f'{scan_name}_{i:03d}')

            else:
                self.save_dataset(X_data, y_data, scan_name)

        self.split_dataset()

    def save_dataset(self, X, y, scan_name):
        if y.max() > 0.0 or not self.only_masks:
            self.save_by_type(X, scan_name, 'images')
            self.save_by_type(y, scan_name, 'labels')

    def save_by_type(self, data, name, types):
        data_full_path = os.path.join(self.out_dataset_dir, types)
        data_full_name = os.path.join(data_full_path, name)

        if not os.path.exists(data_full_path):
            os.makedirs(data_full_path)

        norm_data = utils.norm_to_uint8(data)

        if self.dims == '2d':
            print(f'Saving {types} as {data_full_name}.png')
            im = Image.fromarray(np.rot90(norm_data.squeeze()))
            im.save(f'{data_full_name}.png')
        else:
            print(f'Saving {types} as {data_full_name}.npy')
            np.save(data_full_name, norm_data)

        print('Done.')

    """Returns images and labels with requested format
    Images and labels are resized to desired standard size
    Labels are also binarized based on labels
    """
    def prepare_images_labels(self, scan_name, path) -> (np.ndarray, np.ndarray):
        print(f'Loading from {path}/{self.input_label_niftii}')
        label_data = utils.load_image_data(self.input_label_niftii, path)
        prepared_labels = utils.binarize(label_data, self.labels)
        prepared_labels = utils.resize(prepared_labels, self.scan_shape)

        print(f'Loading from {path}/{self.input_image_niftii}...')
        image_data = utils.load_image_data(self.input_image_niftii, path)
        prepared_images = utils.resize(image_data, self.scan_shape)

        return prepared_images, prepared_labels

    def split_dataset(self):
        X_files = glob.glob(os.path.join(self.out_dataset_dir, 'images', '*.???'))
        y_files = glob.glob(os.path.join(self.out_dataset_dir, 'labels', '*.???'))

        X_train, X_valid, y_train, y_valid = train_test_split(X_files, y_files, test_size=0.2, random_state=1)
        X_valid, X_test, y_valid, y_test =train_test_split(X_valid, y_valid, test_size=0.25, random_state=1)

        files = {
            'images': { 'train': X_train, 'valid': X_valid, 'test': X_test },
            'labels': { 'train': y_train, 'valid': y_valid, 'test': y_test }
        }

        for dataset in ('train', 'valid', 'test'):
            for types in ('images', 'labels'):
                types_dir = os.path.join(self.out_dataset_dir, dataset, types)
                os.makedirs(types_dir)
                for images in files[types][dataset]:
                    shutil.move(images, os.path.join(self.out_dataset_dir, dataset, types))

        shutil.rmtree(os.path.join(self.out_dataset_dir, 'images'))
        shutil.rmtree(os.path.join(self.out_dataset_dir, 'labels'))


    """ Returns files from dataset
    Files are splited by set and proper type (numpy or pngs) are
    chosen based on model (3d or 2d)
    """
    def get_train_valid_test_files(self):
        X = { 'train': list(), 'valid': list(), 'test': list() }
        y = { 'train': list(), 'valid': list(), 'test': list() }

        for dataset in ('train', 'valid', 'test'):
            X[dataset] = sorted(
                glob.glob(
                    os.path.join(self.out_dataset_dir, dataset, 'images', '*.???')
                )[:self.limit]
            )
            y[dataset] = sorted(
                glob.glob(
                    os.path.join(self.out_dataset_dir, dataset, 'labels', '*.???')
                )[:self.limit]
            )

        return X['train'], X['valid'], X['test'], y['train'], y['valid'], y['test']

    
