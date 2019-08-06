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
    depth=cfg.dataset['image_depth']
  ):
    self.arch = arch
    self.checkpoint = checkpoint

    self.model = network.get(
      name=arch,
      loss_function=loss.get(loss_function),
      width=width,
      height=height,
      depth=depth
    )

  def get_model_summary(self):
    self.model.summary()


