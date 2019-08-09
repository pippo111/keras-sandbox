from common import model
from common import dataset

import matplotlib
import matplotlib.pyplot as plt

import config as cfg

matplotlib.use("TkAgg")

my_dataset = dataset.MyDataset(
  is_3d=True
)

my_model = model.MyModel(
  arch='Unet3d',
  checkpoint='unet3d'
)

train_generator, test_generator = my_dataset.create_test_train_gen()
my_model.load()
predicted = my_model.evaluate(test_generator)

test_image, test_mask = test_generator.__getitem__(0)

test_image.shape
image = test_image.squeeze()
image.shape

image_0 = image[55, :, :]
image_1 = image[:, 55, :]
image_2 = image[:, :, 100]

test_mask.shape
mask = test_mask.squeeze()
mask.shape

mask_0 = mask[55, :, :]
mask_1 = mask[:, 55, :]
mask_2 = mask[:, :, 100]

pred = predicted.squeeze()
pred_0 = pred[55, :, :]
pred_1 = pred[:, 55, :]
pred_2 = pred[:, :, 100]

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

plt.show()