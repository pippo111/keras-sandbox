import tensorflow as tf
from keras import backend as K

from keras.optimizers import Adam
from keras_radam import RAdam

def get(name):
  optimizer_fn = dict(
    Adam=Adam(),
    RAdam=RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=1e-5)
  )

  return optimizer_fn[name]
