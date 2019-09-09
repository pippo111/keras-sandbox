from keras import backend as K
import tensorflow as tf

def surface_loss():
    def surface_coef(y_dist, y_pred):
        y_dist_struct = y_dist[..., 1]
        y_pred_struct = y_pred[..., 1]

        multipled = tf.einsum("bwhd,bwhd->bwhd", y_pred_struct, y_dist_struct)

        loss = K.mean(multipled)

        return loss

    return surface_coef
