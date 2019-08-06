from common import dataset

my_dataset = dataset.MyDataset(
  scans = ['NKI-RS-22-1', 'NKI-RS-22-2', 'NKI-RS-22-3'],
  labels = [0.0]
)
my_dataset.create_dataset_3d()
