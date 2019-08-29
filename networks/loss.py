from networks.losses.dice import dice_coef_loss
from networks.losses.wce import create_weighted_binary_crossentropy


def get(name, weights=None):
    loss_fn = dict(
        binary='binary_crossentropy',
        dice=dice_coef_loss,
        wce=create_weighted_binary_crossentropy(weights)
    )

    return loss_fn[name]
