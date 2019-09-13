import numpy as np
import random
from keras.utils import Sequence
from keras.preprocessing.image import load_img, img_to_array

from common.utils import augment_3d, one_hot_encode

class DataSequence3d(Sequence):
    def __init__(self, X_set, y_set, batch_size, shuffle=True, augmentation=False):
        self.X, self.y = X_set, y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_X = np.array([np.load(X) for X in batch_X]).astype(np.float32)
        batch_y = np.array([np.load(y) for y in batch_y]).astype(np.float32)

        if self.augmentation:
            batch_X, batch_y = zip(*[augment_3d(x, y) for x, y in zip(batch_X, batch_y)])
        
        return batch_X, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            to_shuffle = list(zip(self.X, self.y))
            random.shuffle(to_shuffle)
            self.X, self.y = zip(*to_shuffle)

class DataSequence2d(Sequence):
    def __init__(self, X_set, y_set, batch_size, shuffle=True, augmentation=False):
        self.X, self.y = X_set, y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_X = np.stack([np.load(X) for X in batch_X])
        batch_y = np.stack([np.load(y) for y in batch_y])

        return batch_X, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            to_shuffle = list(zip(self.X, self.y))
            random.shuffle(to_shuffle)
            self.X, self.y = zip(*to_shuffle)