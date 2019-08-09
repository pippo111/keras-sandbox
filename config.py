arch = 'ResUnet'
loss_fn = 'dice'
axis = 1

# Dataset setup
dataset = {
  'limit': None,
  'train_dir': 'datasets/z_train_axis{}'.format(axis),
  'validation_dir': 'datasets/z_validation_axis{}'.format(axis),
  'test_dir': 'datasets/z_validation_axis{}'.format(axis),
  'image_width': 44,
  'image_height': 64,
  'image_depth': 64
}

# Model setup
model = {
  'arch': arch,
  'loss_fn': loss_fn,  # binary | dice | wce
  'checkpoint': 'weights.axis_{}_{}_{}'.format(axis, arch, loss_fn),
  'epochs': 50,
  'batch_size': 8,
  'seed': 1,
  'threshold': 0.5,
  'filters': 16
}

# Output result setup
output = {
  # 'slice_numbers': [0,1,2,3,4,5],
  'slice_numbers': [511],
  'save_slice': False
}

# Agrs for data augmentation
generator_args = dict(
  # horizontal_flip=True,
  # vertical_flip=True,
  rotation_range=5,
  width_shift_range=0.1,
  height_shift_range=0.1,
  # zoom_range=0.05
)
