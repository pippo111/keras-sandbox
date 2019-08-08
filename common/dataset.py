import numpy as np
import os
import glob
import nibabel as nib
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom

from common import data_sequence
	
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

def load_image_data(filename, path='.'):
  full_path = os.path.join(path, filename)
  img = nib.load(full_path)
  img_data = img.get_fdata(dtype=np.float32)

  return img_data

class MyDataset():
  def __init__(self,
    scans = [],
    labels = [0.0],
    is_3d = False,
    collection_name = 'mindboggle',
    input_label_niftii = 'aseg-in-t1weighted_2std.nii.gz',
    input_image_niftii = 't1weighted_2std.nii.gz',
    width=176,
    height=256,
    depth=256,
    batch_size = 1,
    limit = 20
  ):
    self.scans = scans
    self.labels = labels
    self.is_3d = is_3d
    self.collection_name = collection_name
    self.input_label_niftii = input_label_niftii
    self.input_image_niftii = input_image_niftii
    self.batch_size = batch_size
    self.width = width
    self.height = height
    self.depth = depth
    self.limit = limit

    self.in_dataset_dir = './input/niftii'
    self.out_dataset_dir = './input/datasets'

  def save_3d(self, data, name, types):
    norm_data = norm_to_uint8(data)
    data_full_path = os.path.join(self.out_dataset_dir, self.collection_name, types)
    data_full_name = os.path.join(data_full_path, name)

    if not os.path.exists(data_full_path):
      os.makedirs(data_full_path)

    print(f'Saving {types} as {data_full_name}.npy')

    im_width, im_height, im_depth = norm_data.shape
    width_ratio = self.width / im_width
    height_ratio = self.height /im_height
    depth_ratio = self.depth / im_depth
    resized_data = zoom(norm_data, (width_ratio, height_ratio, depth_ratio))
    reshaped_data = resized_data.reshape((self.width, self.height, self.depth, 1))

    np.save(data_full_name, reshaped_data)

  def create_image_3d(self, scan_name, path):
    print(f'Loading from {path}/{self.input_image_niftii}...')
    image_data = load_image_data(self.input_image_niftii, path)

    self.save_3d(image_data, scan_name, 'images')

  def create_label_3d(self, scan_name, path):
    print(f'Loading from {path}/{self.input_label_niftii}')
    label_data = load_image_data(self.input_label_niftii, path)
    binary_data = convert_to_binary_3d(label_data, self.labels)

    self.save_3d(binary_data, scan_name, 'labels')

  # Public api
  def create_dataset_3d(self):
    for scan_name in self.scans:
      full_path = os.path.join(self.in_dataset_dir, scan_name)
      self.create_image_3d(scan_name, full_path)
      self.create_label_3d(scan_name, full_path)

  def create_test_train_gen(self):
    if self.is_3d:
      X_files = glob.glob(os.path.join(self.out_dataset_dir, self.collection_name, 'images', '*.npy'))[:self.limit]
      y_files = glob.glob(os.path.join(self.out_dataset_dir, self.collection_name, 'labels', '*.npy'))[:self.limit]
    else:
      X_files = glob.glob(os.path.join(self.out_dataset_dir, self.collection_name, 'images', '*.png'))[:self.limit]
      y_files = glob.glob(os.path.join(self.out_dataset_dir, self.collection_name, 'labels', '*.png'))[:self.limit]

    X_train, X_test, y_train, y_test = train_test_split(X_files, y_files, test_size=0.2, random_state=1)

    train_generator = data_sequence.DataSequence3d(X_train, y_train, self.batch_size)
    test_generator = data_sequence.DataSequence3d(X_test, y_test, self.batch_size)

    self.train_generator = train_generator
    self.test_generator = test_generator

    return train_generator, test_generator
