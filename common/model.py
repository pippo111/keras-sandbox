from common import network
from common import loss
import config as cfg

class MyModel():
  def __init__(self,
    arch,
    checkpoint,
    loss_function=cfg.model['loss_fn'],
    input_cols=cfg.dataset['image_width'],
    input_rows=cfg.dataset['image_height']
  ):
    self.arch = arch
    self.checkpoint = checkpoint

    self.model = network.get(
      name=arch,
      loss_function=loss.get(loss_function),
      input_cols=input_cols,
      input_rows=input_rows
    )

  def get_model_summary(self):
    self.model.summary()


