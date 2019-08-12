from time import time
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from networks import network
from networks import loss
import config as cfg

class MyModel():
  def __init__(self,
    arch,
    checkpoint,
    loss_function,
    width=176,
    height=256,
    depth=256,
    epochs=50,
    filters=16
  ):
    print(width, height, depth)
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

  def load(self):
    self.model.load_weights(f'output/models/{self.checkpoint}.hdf5')

  def train(self, train_generator, test_generator):
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
    
  def evaluate(self, test_generator):
    history = self.model.evaluate_generator(test_generator, verbose=1)
    print(history)

    predicted = self.model.predict_generator(test_generator, verbose=1)
    
    return predicted
