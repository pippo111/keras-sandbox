from networks.losses.dice import dice_loss
from networks.losses.wce import weighted_binary_crossentropy_loss
from networks.losses.surface import surface_loss


def get(name, weights=None):
    loss_fn = dict(
        binary='binary_crossentropy',
        dice=dice_loss,
        wce=weighted_binary_crossentropy_loss(weights),
        surface=surface_loss()
    )

    return loss_fn[name]
