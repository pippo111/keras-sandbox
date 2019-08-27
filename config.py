# Brain 48x64x64 offsets: 27, 26, 42
# Brain 96x128x128 offsets: 45, 44, 84
# Left / right lateral ventricles 48x64x64 offsets: 27, 26, 42
# Left / right lateral ventricles 96x128x128 offsets: 45, 44, 84

# Dataset common setup
dataset = {
    'collection_name': 'mindboggle_84_Nx48x64x64_lateral_ventricle',
    'input_label_niftii': 'aseg-in-t1weighted_2std.nii.gz',
    'input_image_niftii': 't1weighted_2std.nii.gz',
    'scan_shape': (192, 256, 256),
    'input_shape': (48, 64, 64),
    'labels': [4.0, 43.0],
    'limit': None
}

# Models common setup
model = {
    'epochs': 100,
    'seed': 1,
    'threshold': 0.5
}

logs = {
    'axis_0': 10,
    'axis_1': 10,
    'axis_2': 28
}

# Model different parameters
setups = [
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 1, 'filters': 8, 'batch_norm': True,
        'optimizer_fn': 'Adam'
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 4, 'filters': 8, 'batch_norm': True,
        'optimizer_fn': 'Adam'
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 8, 'filters': 8, 'batch_norm': True,
        'optimizer_fn': 'Adam'
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 16, 'filters': 8, 'batch_norm': True,
        'optimizer_fn': 'Adam'
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 32, 'filters': 8, 'batch_norm': True,
        'optimizer_fn': 'Adam'
    },

    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 1, 'filters': 16, 'batch_norm': True,
        'optimizer_fn': 'Adam'
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 4, 'filters': 16, 'batch_norm': True,
        'optimizer_fn': 'Adam'
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 8, 'filters': 16, 'batch_norm': True,
        'optimizer_fn': 'Adam'
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 16, 'filters': 16, 'batch_norm': True,
        'optimizer_fn': 'Adam'
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 32, 'filters': 16, 'batch_norm': True,
        'optimizer_fn': 'Adam'
    }
]

scans = [
    'Afterthought-1',
    'MMRR-3T7T-2-1',
    'MMRR-3T7T-2-2',
    'MMRR-21-1',
    'MMRR-21-2',
    'MMRR-21-3',
    'MMRR-21-4',
    'MMRR-21-5',
    'MMRR-21-6',
    'MMRR-21-7',
    'MMRR-21-8',
    'MMRR-21-9',
    'MMRR-21-10',
    'MMRR-21-11',
    'MMRR-21-12',
    'MMRR-21-13',
    'MMRR-21-14',
    'MMRR-21-15',
    'MMRR-21-16',
    'MMRR-21-17',
    'MMRR-21-18',
    'MMRR-21-19',
    'MMRR-21-20',
    'MMRR-21-21',
    'NKI-RS-22-1',
    'NKI-RS-22-2',
    'NKI-RS-22-3',
    'NKI-RS-22-4',
    'NKI-RS-22-5',
    'NKI-RS-22-6',
    'NKI-RS-22-7',
    'NKI-RS-22-8',
    'NKI-RS-22-9',
    'NKI-RS-22-10',
    'NKI-RS-22-11',
    'NKI-RS-22-12',
    'NKI-RS-22-13',
    'NKI-RS-22-14',
    'NKI-RS-22-15',
    'NKI-RS-22-16',
    'NKI-RS-22-17',
    'NKI-RS-22-18',
    'NKI-RS-22-19',
    'NKI-RS-22-20',
    'NKI-TRT-20-1',
    'NKI-TRT-20-2',
    'NKI-TRT-20-3',
    'NKI-TRT-20-4',
    'NKI-TRT-20-5',
    'NKI-TRT-20-6',
    'NKI-TRT-20-7',
    'NKI-TRT-20-8',
    'NKI-TRT-20-9',
    'NKI-TRT-20-10',
    'NKI-TRT-20-11',
    'NKI-TRT-20-12',
    'NKI-TRT-20-13',
    'NKI-TRT-20-14',
    'NKI-TRT-20-15',
    'NKI-TRT-20-16',
    'NKI-TRT-20-17',
    'NKI-TRT-20-18',
    'NKI-TRT-20-19',
    'NKI-TRT-20-20',
    'OASIS-TRT-20-1',
    'OASIS-TRT-20-2',
    'OASIS-TRT-20-3',
    'OASIS-TRT-20-4',
    'OASIS-TRT-20-5',
    'OASIS-TRT-20-6',
    'OASIS-TRT-20-7',
    'OASIS-TRT-20-8',
    'OASIS-TRT-20-9',
    'OASIS-TRT-20-10',
    'OASIS-TRT-20-11',
    'OASIS-TRT-20-12',
    'OASIS-TRT-20-13',
    'OASIS-TRT-20-14',
    'OASIS-TRT-20-15',
    'OASIS-TRT-20-16',
    'OASIS-TRT-20-17',
    'OASIS-TRT-20-18',
    'OASIS-TRT-20-19',
    'OASIS-TRT-20-20'
  ]
  