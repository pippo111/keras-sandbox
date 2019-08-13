from common import model
from common import dataset
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import config as cfg

matplotlib.use("TkAgg")

my_dataset = dataset.MyDataset(
  collection_name = cfg.dataset['collection_name']
)

my_model = model.MyModel(
  arch = cfg.model['arch'],
  loss_function = cfg.model['loss_fn'],
  batch_norm=cfg.model['batch_norm'],
  checkpoint = cfg.model['checkpoint'],
  width = cfg.dataset['width'],
  height = cfg.dataset['height'],
  depth = cfg.dataset['depth'],
  epochs = cfg.model['epochs'],
  filters = cfg.model['filters']
)

train_generator, test_generator = my_dataset.create_train_test_gen()
my_model.load()
predicted = my_model.evaluate(test_generator)
preds_bin = (predicted > 0.5).astype(np.uint8)

test_image, test_mask = test_generator.__getitem__(0)

test_image.shape
image = test_image[0].squeeze()
print(image.shape)

image_0 = image[25, :, :]
image_1 = image[:, 20, :]
image_2 = image[:, :, 25]

test_mask.shape
mask = test_mask[0].squeeze()
print(mask.shape)

mask_0 = mask[25, :, :]
mask_1 = mask[:, 20, :]
mask_2 = mask[:, :, 25]

pred = preds_bin[0].squeeze()
pred_0 = pred[25, :, :]
pred_1 = pred[:, 20, :]
pred_2 = pred[:, :, 25]

fig, ax = plt.subplots(3, 3, figsize=(20, 20))

ax[0][0].imshow(image_0, cmap='gray')
ax[0][1].imshow(image_1, cmap='gray')
ax[0][2].imshow(image_2, cmap='gray')

ax[1][0].imshow(mask_0, cmap='gray')
ax[1][1].imshow(mask_1, cmap='gray')
ax[1][2].imshow(mask_2, cmap='gray')

ax[2][0].imshow(pred_0, cmap='gray')
ax[2][1].imshow(pred_1, cmap='gray')
ax[2][2].imshow(pred_2, cmap='gray')

fig.savefig(f'output/models/{cfg.model["checkpoint"]}.png')