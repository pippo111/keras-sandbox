from networks.losses.dice import dice_coef_loss
from networks.losses.wce import weighted_binary_crossentropy_loss


def get(name, weights=None):
    loss_fn = dict(
        binary='binary_crossentropy',
        dice=dice_coef_loss,
        wce=weighted_binary_crossentropy_loss
    )

    return loss_fn[name]
