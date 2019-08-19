from time import time
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import numpy as np

from common.time_callback import TimeHistory
from common.reinit_falsestart_callback import ReinitWeightOnFalseStart
from networks import network
from networks import loss
import config as cfg

class MyModel():
  def __init__(self,
    arch,
    checkpoint,
    loss_function,
    batch_norm=False,
    threshold=0.5,
    width=176,
    height=256,
    depth=256,
    epochs=50,
    filters=16
  ):
    self.arch = arch
    self.checkpoint = checkpoint
    self.epochs = epochs

    self.model = network.get(
      name=arch,
      loss_function=loss.get(loss_function),
      batch_norm=batch_norm,
      width=width,
      height=height,
      depth=depth,
      n_filters=filters
    )

  def get_model_summary(self):
    self.model.summary()

  def load(self):
    self.model.load_weights(f'output/models/{self.checkpoint}.hdf5')

  def train(self, train_generator, test_generator):
    time_callback = TimeHistory()

    history = self.model.fit_generator(
      train_generator,
      epochs=self.epochs,
      callbacks=[
        time_callback,
        ReinitWeightOnFalseStart(patience=3, trials=1, checks=10, verbose=1),
        # TensorBoard(log_dir=f'output/logs/{time()}-{self.checkpoint}'),
        # EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(f'output/models/{self.checkpoint}.hdf5', verbose=1, save_best_only=True, save_weights_only=True)
      ],
      validation_data=test_generator
    )

    times = time_callback.times
    epoch_time = int(np.mean(times))

    return history, epoch_time
    
  def evaluate(self, test_generator):
    history = self.model.evaluate_generator(test_generator, verbose=1)
    
    return history

  def predict(self, test_generator):
    X_preds = self.model.predict_generator(test_generator, verbose=1)
    y_preds = (X_preds > 0.5).astype(np.uint8)
    
    return X_preds, y_preds
