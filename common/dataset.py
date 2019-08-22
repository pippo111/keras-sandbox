import os
import numpy as np
import glob
from sklearn.model_selection import train_test_split

from common import data_sequence
from common import utils

class MyDataset():
  def __init__(self,
    scans = [],
    labels = [255.0],
    collection_name = 'mindboggle',
    input_label_niftii = 'aseg-in-t1weighted_2std.nii.gz',
    input_image_niftii = 't1weighted_2std.nii.gz',
    width=48,
    height=64,
    depth=64,
    slice_depth=8,
    batch_size = 32,
    limit = None
  ):
    self.scans = scans
    self.labels = labels
    self.collection_name = collection_name
    self.input_label_niftii = input_label_niftii
    self.input_image_niftii = input_image_niftii
    self.batch_size = batch_size
    self.width = width
    self.height = height
    self.depth = depth
    self.slice_depth = slice_depth
    self.limit = limit

    self.in_dataset_dir = './input/niftii'
    self.out_dataset_dir = './input/datasets'

  def save_3d(self, data, name, types):
    data_full_path = os.path.join(self.out_dataset_dir, self.collection_name, types)
    data_full_name = os.path.join(data_full_path, name)

    if not os.path.exists(data_full_path):
      os.makedirs(data_full_path)

    print(f'Saving {types} as {data_full_name}.npy')

    norm_data = utils.norm_to_uint8(data)

    # save normalized to uint (0 - 255)
    np.save(data_full_name, norm_data)
    print('Done.')

  def create_image_3d(self, scan_name, path):
    print(f'Loading from {path}/{self.input_image_niftii}...')
    image_data = utils.load_image_data(self.input_image_niftii, path)
    prepared_data = utils.resize_3d(image_data, self.width, self.height, self.depth)

    if self.slice_depth:
      for slice_no, i in enumerate(np.arange(0, self.depth, self.slice_depth)):
        sliced_data = prepared_data[:,:,i : i + self.slice_depth]
        self.save_3d(sliced_data, f'{scan_name}_{slice_no}', 'images')
    else:
      self.save_3d(prepared_data, scan_name, 'images')


  def create_label_3d(self, scan_name, path):
    print(f'Loading from {path}/{self.input_label_niftii}')
    label_data = utils.load_image_data(self.input_label_niftii, path)
    prepared_data = utils.convert_to_binary_3d(label_data, self.labels)
    prepared_data = utils.resize_3d(prepared_data, self.width, self.height, self.depth)
    
    if self.slice_depth:
      for slice_no, i in enumerate(np.arange(0, self.depth, self.slice_depth)):
        sliced_data = prepared_data[:,:,i : i + self.slice_depth]
        self.save_3d(sliced_data, f'{scan_name}_{slice_no}', 'labels')
    else:
      self.save_3d(prepared_data, scan_name, 'labels')

  # Public api
  def create_dataset_3d(self):
    for scan_name in self.scans:
      full_path = os.path.join(self.in_dataset_dir, scan_name)
      self.create_image_3d(scan_name, full_path)
      self.create_label_3d(scan_name, full_path)

  def create_train_test_gen(self):
    X_files = glob.glob(os.path.join(self.out_dataset_dir, self.collection_name, 'images', '*.npy'))[:self.limit]
    y_files = glob.glob(os.path.join(self.out_dataset_dir, self.collection_name, 'labels', '*.npy'))[:self.limit]

    X_train, X_test, y_train, y_test = train_test_split(X_files, y_files, test_size=0.2, random_state=1)

    train_generator = data_sequence.DataSequence3d(X_train, y_train, self.batch_size, augmentation=True)
    test_generator = data_sequence.DataSequence3d(X_test, y_test, self.batch_size, shuffle=False, augmentation=False)

    self.train_generator = train_generator
    self.test_generator = test_generator

    return train_generator, test_generator

  def get_count(self):
    X_files = glob.glob(os.path.join(self.out_dataset_dir, self.collection_name, 'images', '*.npy'))[:self.limit]
    return len(X_files)