import numpy as np
import random
from keras.utils import Sequence

class DataSequence3d(Sequence):

  def __init__(self, X_set, y_set, batch_size):
    self.X, self.y = X_set, y_set
    self.batch_size = batch_size
    self.on_epoch_end()

  def __len__(self):
    return int(np.ceil(len(self.X) / float(self.batch_size)))

  def __getitem__(self, idx):
    batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
    batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
    
    batch_X = [np.load(X) for X in batch_X]
    batch_y = [np.load(y) for y in batch_y]
    
    return np.array(batch_X), np.array(batch_y)
  
  def on_epoch_end(self):   
    to_shuffle = list(zip(self.X, self.y))
    random.shuffle(to_shuffle)
    self.X, self.y = zip(*to_shuffle)