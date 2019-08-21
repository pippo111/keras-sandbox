from keras.optimizers import Adam

from networks.arch.unet import unet
from networks.arch.unet import unet_bn
from networks.arch.unet3d import unet3d
from networks.arch.resunet import resunet
from networks.arch.resunet3d import resunet3d

def get(
    name,
    width=176,
    height=256,
    depth=1,
    n_filters=16,
    loss_function='binary_crossentropy',
    optimizer_function=Adam(),
    batch_norm=False
):
    networks = dict(
        Unet=unet,
        UnetBN=unet_bn,
        ResUnet=resunet,
        Unet3d=unet3d,
        ResUnet3d=resunet3d
    )
    
    return networks[name](width, height, depth, n_filters, loss_function, optimizer_function, batch_norm)
