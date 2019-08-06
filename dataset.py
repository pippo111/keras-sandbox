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

def convert_to_binary_3d(data, labels):
  binary_data = np.array([[[0.0 if pixel in labels else 255.0 for pixel in row] for row in matrix] for matrix in data]).astype(np.float32)
  return binary_data

def load_image_data(filename, path='.'):
  full_path = os.path.join(path, filename)
  img = nib.load(full_path)
  img_data = img.get_fdata(dtype=np.float32)

  return img_data

class MyDataset():
  def __init__(self,
    scans,
    labels,
    collection_name = 'mindboggle',
    input_label_niftii = 'aseg-in-t1weighted_2std.nii.gz',
    input_image_niftii = 't1weighted_2std.nii.gz'
  ):
    self.scans = scans
    self.labels = labels
    self.collection_name = collection_name
    self.input_label_niftii = input_label_niftii
    self.input_image_niftii = input_image_niftii

  def save_3d(self, data, name, types):
    norm_data = norm_to_uint8(data)
    data_full_path = os.path.join('./datasets', self.collection_name, types)
    data_full_name = os.path.join(data_full_path, name)

    if not os.path.exists(data_full_path):
      os.makedirs(data_full_path)

    print(f'Saving {types} as {data_full_name}.npy')
    np.save(data_full_name, norm_data)

  def create_image_3d(self, scan_name, path):
    print(f'Loading from {path}/{self.input_image_niftii}...')
    image_data = load_image_data(self.input_image_niftii, path)

    self.save_3d(image_data, scan_name, 'images')

  def create_label_3d(self, scan_name, path):
    print(f'Loading from {path}/{self.input_label_niftii}')
    label_data = load_image_data(self.input_label_niftii, path)
    binary_data = convert_to_binary_3d(label_data, self.labels)

    self.save_3d(binary_data, scan_name, 'labels')

  def create_dataset_3d(self):
    for scan_name in self.scans:
      full_path = os.path.join('./niftii', scan_name)
      self.create_image_3d(scan_name, full_path)
      self.create_label_3d(scan_name, full_path)