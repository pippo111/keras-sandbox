from networks.losses.dice import dice_coef_loss
from networks.losses.wce import weighted_cross_entropy


def get(name):
    loss_fn = dict(
        binary='binary_crossentropy',
        dice=dice_coef_loss,
        wce=weighted_cross_entropy(0.5)
    )

    return loss_fn[name]
