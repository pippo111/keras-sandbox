import numpy as np

from common import model
from common import dataset
from common.utils import get_all_gen_items

import matplotlib
import matplotlib.pyplot as plt

import config as cfg

my_dataset = dataset.MyDataset(
    collection_name=cfg.dataset['collection_name']
)

train_generator, valid_generator, test_generator = my_dataset.create_train_valid_test_gen()
test_image, test_mask = get_all_gen_items(test_generator)
test_image, test_mask = test_generator.__getitem__(0)

image = test_image[0].squeeze()

image_0 = image[cfg.logs['axis_0'], :, :]
image_1 = image[:, cfg.logs['axis_1'], :]
image_2 = image[:, :, cfg.logs['axis_2']]

image.shape

mask = test_mask[0].squeeze()

width, height, depth = cfg.dataset['input_shape']

for i in range(width):
    for j in range(height):
        for k in range(depth):
            # print(mask[i,j,k].max())
            if mask[i,j,k] > 0:
                print(i,j,k, mask[i,j,k])

mask_0 = mask[cfg.logs['axis_0'], :, :]
mask_1 = mask[:, cfg.logs['axis_1'], :]
mask_2 = mask[:, :, cfg.logs['axis_2']]

mask.shape

fig, ax = plt.subplots(2, 3, figsize=(20, 20))

ax[0][0].imshow(np.rot90(image_0), cmap='gray')
ax[0][1].imshow(np.rot90(image_1), cmap='gray')
ax[0][2].imshow(np.rot90(image_2), cmap='gray')

ax[1][0].imshow(np.rot90(mask_0), cmap='gray')
ax[1][1].imshow(np.rot90(mask_1), cmap='gray')
ax[1][2].imshow(np.rot90(mask_2), cmap='gray')

plt.show()
