import config as cfg

from common import dataset

my_dataset = dataset.MyDataset(
  collection_name = cfg.dataset['collection_name'],
  input_label_niftii = cfg.dataset['input_label_niftii'],
  input_image_niftii = cfg.dataset['input_image_niftii'],
  labels = cfg.dataset['labels'],
  scan_shape=cfg.dataset['scan_shape'],
  input_shape=cfg.dataset['input_shape'],
  scans=cfg.scans
)
my_dataset.create_dataset_3d()
