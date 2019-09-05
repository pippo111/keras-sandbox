from keras import backend as K
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def weighted_binary_crossentropy_loss():
    def weighted_binary_crossentropy_coef(y_true, y_pred):
        # Calculate the weights
        y_flat = y_true.flatten()
        class_weights = compute_class_weight('balanced', np.unique(y_flat), y_flat)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * class_weights[1] + (1. - y_true) * class_weights[0]
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy_coef
