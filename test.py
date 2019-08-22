#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

from common import model
from common import dataset

import matplotlib
import matplotlib.pyplot as plt

import config as cfg


# In[2]:


my_dataset = dataset.MyDataset(
    collection_name=cfg.dataset['collection_name']
)


# In[3]:


train_generator, test_generator = my_dataset.create_train_test_gen()
test_image, test_mask = test_generator.__getitem__(4)


# In[4]:


image = test_image[0].squeeze()


image_0 = image[45, :, :]
image_1 = image[:, 44, :]
image_2 = image[:, :, 2]

image.shape


# In[5]:


mask = test_mask[0].squeeze()


mask_0 = mask[45, :, :]
mask_1 = mask[:, 44, :]
mask_2 = mask[:, :, 2]

mask.shape


# In[6]:


fig, ax = plt.subplots(2, 3, figsize=(20, 20))

ax[0][0].imshow(np.rot90(image_0), cmap='gray')
ax[0][1].imshow(np.rot90(image_1), cmap='gray')
ax[0][2].imshow(np.rot90(image_2), cmap='gray')

ax[1][0].imshow(np.rot90(mask_0), cmap='gray')
ax[1][1].imshow(np.rot90(mask_1), cmap='gray')
ax[1][2].imshow(np.rot90(mask_2), cmap='gray')

plt.show()
# In[ ]:



