from common import model
from common import dataset

import config as cfg

my_dataset = dataset.MyDataset(
  collection_name = cfg.dataset['collection_name']
)

my_model = model.MyModel(
  arch = cfg.model['arch'],
  loss_function = cfg.model['loss_fn'],
  checkpoint = cfg.model['checkpoint'],
  width = cfg.dataset['width'],
  height = cfg.dataset['height'],
  depth = cfg.dataset['depth'],
  epochs = cfg.model['epochs'],
  filters = cfg.model['filters']
)

train_generator, test_generator = my_dataset.create_test_train_gen()
my_model.get_model_summary()
my_model.train(train_generator, test_generator)
