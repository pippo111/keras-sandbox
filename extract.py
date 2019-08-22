import config as cfg

from common import dataset

my_dataset = dataset.MyDataset(
  collection_name = cfg.dataset['collection_name'],
  input_label_niftii = cfg.dataset['input_label_niftii'],
  input_image_niftii = cfg.dataset['input_image_niftii'],
  labels = cfg.dataset['labels'],
  width=cfg.dataset['width'],
  height=cfg.dataset['height'],
  depth=cfg.dataset['depth'],
  slice_depth=cfg.dataset['slice_depth'],
  scans=cfg.scans
)
my_dataset.create_dataset_3d()
