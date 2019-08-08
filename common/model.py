from time import time
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from common import network
from common import loss
import config as cfg

class MyModel():
  def __init__(self,
    arch,
    checkpoint,
    loss_function=cfg.model['loss_fn'],
    width=cfg.dataset['image_width'],
    height=cfg.dataset['image_height'],
    depth=cfg.dataset['image_depth'],
    epochs=cfg.model['epochs'],
    filters=cfg.model['filters']
  ):
    self.arch = arch
    self.checkpoint = checkpoint
    self.epochs = epochs

    self.model = network.get(
      name=arch,
      loss_function=loss.get(loss_function),
      width=width,
      height=height,
      depth=depth,
      n_filters=filters
    )

  def get_model_summary(self):
    self.model.summary()

  def start(self, train_generator, test_generator):
    self.model.fit_generator(
      train_generator,
      epochs=self.epochs,
      callbacks=[
        TensorBoard(log_dir=f'output/logs/{time()}-{self.checkpoint}'),
        EarlyStopping(patience=6, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(f'output/models/{self.checkpoint}.hdf5', verbose=1, save_best_only=True, save_weights_only=True)
      ],
      validation_data=test_generator
    )
