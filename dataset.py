import numpy as np
import os
import nibabel as nib

def norm_to_uint8(data):
  max_value = data.max()
  if not max_value == 0:
    data = data / max_value

  data = 255 * data
  img = data.astype(np.uint8)
  return img

def convert_to_binary_mask(data, labels):
  binary_data = np.array([[[255.0 if pixel in labels else 0.0 for pixel in row] for row in matrix] for matrix in data]).astype(np.float32)
  return binary_data

def load_image_data(filename, path='.'):
  full_path = os.path.join(path, filename)
  img = nib.load(full_path)
  img_data = img.get_fdata(dtype=np.float32)

  return img_data

class MyDataset():
  def __init__(self,
    scan_name,
    labels,
    scan_collection_dir = 'niftii',
    collection_name = 'mindboggle',
    input_label_niftii = 'aseg-in-t1weighted_2std.nii.gz',
    input_image_niftii = 't1weighted_2std.nii.gz'
  ):
    self.scan_name = scan_name
    self.labels = labels
    self.collection_name = collection_name
    self.input_label_niftii = input_label_niftii
    self.input_image_niftii = input_image_niftii

    self.scan_full_path = os.path.join('./niftii', self.scan_name)

  def save_3d(self, data, types):
    norm_data = norm_to_uint8(data)
    data_full_path = os.path.join('./datasets', self.collection_name, types)
    data_full_name = os.path.join(data_full_path, self.scan_name)

    if not os.path.exists(data_full_path):
      os.makedirs(data_full_path)

    print(f'Saving {types} into {data_full_name}.npy...')
    np.save(data_full_name, norm_data)

  def create_image_3d(self):
    image_data = load_image_data(self.input_image_niftii, path=self.scan_full_path)
    print(f'Loading from {self.scan_full_path}/{self.input_image_niftii}...')

    self.save_3d(image_data, 'images')

  def create_label_3d(self):
    label_data = load_image_data(self.input_label_niftii, path=self.scan_full_path)
    binary_data = convert_to_binary_mask(label_data, self.labels)

    self.save_3d(binary_data, 'labels')

  def create_dataset_3d(self):
    self.create_image_3d()
    self.create_label_3d()