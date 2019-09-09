import os
import shutil
import numpy as np
import glob
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

    def save_dataset_3d(self, X, y, scan_name):
        self.save_3d(X, scan_name, 'images')
        self.save_3d(y, scan_name, 'labels')

    def save_3d(self, data, name, types):
        data_full_path = os.path.join(self.out_dataset_dir, types)
        data_full_name = os.path.join(data_full_path, name)

        if not os.path.exists(data_full_path):
            os.makedirs(data_full_path)
        
        print(f'Saving {types} as {data_full_name}.npy')

        norm_data = utils.norm_to_uint8(data)

        # save normalized to uint (0 - 255)
        np.save(data_full_name, norm_data)
        print('Done.')

    def prep_images_labels_data(self, scan_name, path):
        print(f'Loading from {path}/{self.input_label_niftii}')
        label_data = utils.load_image_data(self.input_label_niftii, path)
        prepared_labels = utils.convert_to_binary_3d(label_data, self.labels)
        prepared_labels = utils.resize_3d(prepared_labels, self.scan_shape)

        print(f'Loading from {path}/{self.input_image_niftii}...')
        image_data = utils.load_image_data(self.input_image_niftii, path)
        prepared_images = utils.resize_3d(image_data, self.scan_shape)

        return prepared_images, prepared_labels

    # def move_files()

    def split_dataset(self):
        X_files = glob.glob(os.path.join(self.out_dataset_dir, 'images', '*.npy'))
        y_files = glob.glob(os.path.join(self.out_dataset_dir, 'labels', '*.npy'))

        X_train, X_valid, y_train, y_valid = train_test_split(X_files, y_files, test_size=0.2, random_state=1)
        X_valid, X_test, y_valid, y_test =train_test_split(X_valid, y_valid, test_size=0.25, random_state=1)

        X_train_dir = os.path.join(self.out_dataset_dir, 'train', 'images')
        os.makedirs(X_train_dir)
        y_train_dir = os.path.join(self.out_dataset_dir, 'train', 'labels')
        os.makedirs(y_train_dir)
        X_valid_dir = os.path.join(self.out_dataset_dir, 'valid', 'images')
        os.makedirs(X_valid_dir)
        y_valid_dir = os.path.join(self.out_dataset_dir, 'valid', 'labels')
        os.makedirs(y_valid_dir)
        X_test_dir = os.path.join(self.out_dataset_dir, 'test', 'images')
        os.makedirs(X_test_dir)
        y_test_dir = os.path.join(self.out_dataset_dir, 'test', 'labels')
        os.makedirs(y_test_dir)

        for images in X_train:
            shutil.move(images, os.path.join(self.out_dataset_dir, 'train', 'images'))
        for labels in y_train:
            shutil.move(labels, os.path.join(self.out_dataset_dir, 'train', 'labels'))
        for images in X_valid:
            shutil.move(images, os.path.join(self.out_dataset_dir, 'valid', 'images'))
        for labels in y_valid:
            shutil.move(labels, os.path.join(self.out_dataset_dir, 'valid', 'labels'))
        for images in X_test:
            shutil.move(images, os.path.join(self.out_dataset_dir, 'test', 'images'))
        for labels in y_test:
            shutil.move(labels, os.path.join(self.out_dataset_dir, 'test', 'labels'))

        shutil.rmtree(os.path.join(self.out_dataset_dir, 'images'))
        shutil.rmtree(os.path.join(self.out_dataset_dir, 'labels'))

    # Public api
    def create_dataset_3d(self):
        if os.path.exists(self.out_dataset_dir):
            shutil.rmtree(self.out_dataset_dir)
        os.makedirs(self.out_dataset_dir)

        to_slice = self.scan_shape != self.input_shape

        for scan_name in self.scans:
            full_path = os.path.join(self.in_dataset_dir, scan_name)
            X_data, y_data = self.prep_images_labels_data(scan_name, full_path)

            if to_slice:
                sliced_images = utils.slice_3d(X_data, self.input_shape)
                sliced_labels = utils.slice_3d(y_data, self.input_shape)

                for i, labels in enumerate(sliced_labels):
                    if self.only_masks:
                        if labels.max() > 0.0:
                            self.save_dataset_3d(sliced_images[i], labels, f'{scan_name}_{i}')
                    else:
                        self.save_dataset_3d(sliced_images[i], labels, f'{scan_name}_{i}')

            else:
                self.save_dataset_3d(X_data, y_data, scan_name)

        self.split_dataset()

    def get_train_valid_test_files(self):
        X = {
            'train': list(),
            'valid': list(),
            'test': list()
        }

        y = {
            'train': list(),
            'valid': list(),
            'test': list()
        }

        for dataset in ('train', 'valid', 'test'):
            X[dataset] = sorted(
                glob.glob(
                    os.path.join(self.out_dataset_dir, dataset, 'images', '*.npy')
                )[:self.limit]
            )
            y[dataset] = sorted(
                glob.glob(
                    os.path.join(self.out_dataset_dir, dataset, 'labels', '*.npy')
                )[:self.limit]
            )

        return X['train'], X['valid'], X['test'], y['train'], y['valid'], y['test']

    def create_train_valid_test_gen(self):
        X_train, X_valid, X_test, y_train, y_valid, y_test = self.get_train_valid_test_files()

        train_generator = data_sequence.DataSequence3d(X_train, y_train, self.batch_size, augmentation=True, dist=True)
        valid_generator = data_sequence.DataSequence3d(X_valid, y_valid, self.batch_size, shuffle=False, augmentation=False, dist=True)
        test_generator = data_sequence.DataSequence3d(X_test, y_test, self.batch_size, shuffle=False, augmentation=False, dist=False)

        self.train_generator = train_generator
        self.valid_generator = valid_generator
        self.test_generator = test_generator

        return train_generator, valid_generator, test_generator

    def get_count(self):
        X_files = glob.glob(os.path.join(self.out_dataset_dir, 'images', '*.npy'))[:self.limit]
        return len(X_files)