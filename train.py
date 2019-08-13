from common import model
from common import dataset

import config as cfg

my_dataset = dataset.MyDataset(
  collection_name = cfg.dataset['collection_name'],
  batch_size=cfg.model['batch_size']
)

my_model = model.MyModel(
  arch = cfg.model['arch'],
  loss_function = cfg.model['loss_fn'],
  batch_norm=cfg.model['batch_norm'],
  checkpoint = cfg.model['checkpoint'],
  threshold=cfg.model['threshold'],
  width = cfg.dataset['width'],
  height = cfg.dataset['height'],
  depth = cfg.dataset['depth'],
  epochs = cfg.model['epochs'],
  filters = cfg.model['filters']
)

train_generator, test_generator = my_dataset.create_train_test_gen()
my_model.get_model_summary()
history = my_model.train(train_generator, test_generator)

val_loss = history.history['val_loss']
val_acc = history.history['val_acc']
