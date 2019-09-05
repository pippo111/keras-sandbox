from keras import backend as K
import tensorflow as tf

def surface_loss():
    def surface_coef(y_dist, y_pred):
        multipled = tf.einsum("bwhdc,bwhdc->bwhdc", y_pred, y_dist)

        loss = K.mean(multipled)

        return loss

    return surface_coef
