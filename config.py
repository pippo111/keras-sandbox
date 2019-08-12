arch = 'Unet3d'
loss_fn = 'dice'
axis = 1

# Dataset setup
dataset = {
  'collection_name': 'mindboggle_scaled',
  'input_label_niftii': 'aseg-in-t1weighted_2std.nii.gz',
  'input_image_niftii': 't1weighted_2std.nii.gz',
  'width': 48,
  'height': 64,
  'depth': 64,
  'labels': [0.0],
  'batch_size': 8,
  'limit': None
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
  'filters': 8
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
