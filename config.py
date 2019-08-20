arch = 'ResUnet3d'
loss_fn = 'binary'
batch_size = 8
filters = 8
batch_norm = True
checkpoint = f'{arch}_{loss_fn}_f{filters}_bs{batch_size}_bn{batch_norm}'

# Brain 48x64x64 offsets: 27, 26, 42
# Brain 96x128x128 offsets: 45, 44, 84
# Left / right lateral ventricles 48x64x64 offsets: 27, 26, 42
# Left / right lateral ventricles 96x128x128 offsets: 45, 44, 84

# Dataset setup
dataset = {
  'collection_name': 'mindboggle_84_64x64x48_brain',
  'input_label_niftii': 'aseg-in-t1weighted_2std.nii.gz',
  'input_image_niftii': 't1weighted_2std.nii.gz',
  'width': 48,
  'height': 64,
  'depth': 64,
  'labels': [4.0, 43.0],
  'limit': None
}

# Model setup
model = {
  'arch': arch,
  'loss_fn': loss_fn,  # binary | dice | wce
  'checkpoint': checkpoint,
  'epochs': 1,
  'batch_size': batch_size,
  'seed': 1,
  'threshold': 0.5,
  'filters': filters,
  'batch_norm': batch_norm
}
