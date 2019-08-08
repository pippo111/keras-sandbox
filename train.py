from common import model
from common import dataset

import config as cfg

my_dataset = dataset.MyDataset(
  is_3d=True
)

my_model = model.MyModel(
  arch='Unet3d',
  loss_function='dice',
  checkpoint='unet3d_dice'
)

train_generator, test_generator = my_dataset.create_test_train_gen()
my_model.get_model_summary()
my_model.train(train_generator, test_generator)
