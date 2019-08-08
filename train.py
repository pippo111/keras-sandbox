from common import model
from common import dataset

import config as cfg

my_dataset = dataset.MyDataset(
  scans = [
    'NKI-RS-22-1',
    'NKI-RS-22-2',
    'NKI-RS-22-3',
    'NKI-RS-22-4',
    'NKI-RS-22-5',
    'NKI-RS-22-6',
    'NKI-RS-22-7',
    'NKI-RS-22-8',
    'NKI-RS-22-9',
    'NKI-RS-22-10',
    'NKI-RS-22-11',
    'NKI-RS-22-12',
    'NKI-RS-22-13',
    'NKI-RS-22-14',
    'NKI-RS-22-15',
    'NKI-RS-22-16',
    'NKI-RS-22-17',
    'NKI-RS-22-18',
    'NKI-RS-22-19',
    'NKI-RS-22-20',
    'NKI-TRT-20-1',
    'NKI-TRT-20-2',
    'NKI-TRT-20-3',
    'NKI-TRT-20-4',
    'NKI-TRT-20-5',
    'NKI-TRT-20-6',
    'NKI-TRT-20-7',
    'NKI-TRT-20-8',
    'NKI-TRT-20-9',
    'NKI-TRT-20-10',
    'NKI-TRT-20-11',
    'NKI-TRT-20-12',
    'NKI-TRT-20-13',
    'NKI-TRT-20-14',
    'NKI-TRT-20-15',
    'NKI-TRT-20-16',
    'NKI-TRT-20-17',
    'NKI-TRT-20-18',
    'NKI-TRT-20-19',
    'NKI-TRT-20-20',
    'OASIS-TRT-20-1',
    'OASIS-TRT-20-2',
    'OASIS-TRT-20-3',
    'OASIS-TRT-20-4',
    'OASIS-TRT-20-5',
    'OASIS-TRT-20-6',
    'OASIS-TRT-20-7',
    'OASIS-TRT-20-8',
    'OASIS-TRT-20-9',
    'OASIS-TRT-20-10',
    'OASIS-TRT-20-11',
    'OASIS-TRT-20-12',
    'OASIS-TRT-20-13',
    'OASIS-TRT-20-14',
    'OASIS-TRT-20-15',
    'OASIS-TRT-20-16',
    'OASIS-TRT-20-17',
    'OASIS-TRT-20-18',
    'OASIS-TRT-20-19',
    'OASIS-TRT-20-20'
  ],
  is_3d=True
)

my_model = model.MyModel(
  arch='Unet3d',
  checkpoint='unet_3d'
)

train_generator, test_generator = my_dataset.get_test_train_gen()
my_model.get_model_summary()
my_model.start(train_generator, test_generator)
