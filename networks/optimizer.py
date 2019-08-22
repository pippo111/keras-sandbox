import tensorflow as tf
from keras import backend as K

from keras.optimizers import Adam
from keras_radam import RAdam

def get(name):
  optimizer_fn = dict(
    Adam=Adam(),
    RAdam=RAdam()
  )

  return optimizer_fn[name]
