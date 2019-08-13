import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from common import model
from common import dataset
import config as cfg

setups = [
    {
        'arch': 'ResUnet3d', 'loss_fn': 'binary',
        'batch_size': 1, 'filters': 8, 'batch_norm': False
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'binary',
        'batch_size': 1, 'filters': 8, 'batch_norm': True
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'dice',
        'batch_size': 1, 'filters': 8, 'batch_norm': False
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'dice',
        'batch_size': 1, 'filters': 8, 'batch_norm': True
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 1, 'filters': 8, 'batch_norm': False
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 1, 'filters': 8, 'batch_norm': True
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'dice',
        'batch_size': 1, 'filters': 8, 'batch_norm': False
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'dice',
        'batch_size': 1, 'filters': 8, 'batch_norm': True
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'binary',
        'batch_size': 4, 'filters': 8, 'batch_norm': False
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'binary',
        'batch_size': 4, 'filters': 8, 'batch_norm': True
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'dice',
        'batch_size': 4, 'filters': 8, 'batch_norm': False
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'dice',
        'batch_size': 4, 'filters': 8, 'batch_norm': True
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 4, 'filters': 8, 'batch_norm': False
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 4, 'filters': 8, 'batch_norm': True
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'dice',
        'batch_size': 4, 'filters': 8, 'batch_norm': False
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'dice',
        'batch_size': 4, 'filters': 8, 'batch_norm': True
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'binary',
        'batch_size': 8, 'filters': 8, 'batch_norm': False
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'binary',
        'batch_size': 8, 'filters': 8, 'batch_norm': True
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'dice',
        'batch_size': 8, 'filters': 8, 'batch_norm': False
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'dice',
        'batch_size': 8, 'filters': 8, 'batch_norm': True
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 8, 'filters': 8, 'batch_norm': False
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 8, 'filters': 8, 'batch_norm': True
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'dice',
        'batch_size': 8, 'filters': 8, 'batch_norm': False
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'dice',
        'batch_size': 8, 'filters': 8, 'batch_norm': True
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'binary',
        'batch_size': 1, 'filters': 16, 'batch_norm': False
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'binary',
        'batch_size': 1, 'filters': 16, 'batch_norm': True
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'dice',
        'batch_size': 1, 'filters': 16, 'batch_norm': False
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'dice',
        'batch_size': 1, 'filters': 16, 'batch_norm': True
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 1, 'filters': 16, 'batch_norm': False
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 1, 'filters': 16, 'batch_norm': True
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'dice',
        'batch_size': 1, 'filters': 16, 'batch_norm': False
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'dice',
        'batch_size': 1, 'filters': 16, 'batch_norm': True
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'binary',
        'batch_size': 4, 'filters': 16, 'batch_norm': False
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'binary',
        'batch_size': 4, 'filters': 16, 'batch_norm': True
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'dice',
        'batch_size': 4, 'filters': 16, 'batch_norm': False
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'dice',
        'batch_size': 4, 'filters': 16, 'batch_norm': True
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 4, 'filters': 16, 'batch_norm': False
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 4, 'filters': 16, 'batch_norm': True
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'dice',
        'batch_size': 4, 'filters': 16, 'batch_norm': False
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'dice',
        'batch_size': 4, 'filters': 16, 'batch_norm': True
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'binary',
        'batch_size': 8, 'filters': 16, 'batch_norm': False
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'binary',
        'batch_size': 8, 'filters': 16, 'batch_norm': True
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'dice',
        'batch_size': 8, 'filters': 16, 'batch_norm': False
    },
    {
        'arch': 'ResUnet3d', 'loss_fn': 'dice',
        'batch_size': 8, 'filters': 16, 'batch_norm': True
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 8, 'filters': 16, 'batch_norm': False
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'binary',
        'batch_size': 8, 'filters': 16, 'batch_norm': True
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'dice',
        'batch_size': 8, 'filters': 16, 'batch_norm': False
    },
    {
        'arch': 'Unet3d', 'loss_fn': 'dice',
        'batch_size': 8, 'filters': 16, 'batch_norm': True
    }
]

params = {
    'arch': pd.Series(),
    'loss_fn': pd.Series(),
    'batch_size': pd.Series(),
    'filters': pd.Series(),
    'batch_norm': pd.Series(),
    'val_loss': pd.Series(),
    'val_acc': pd.Series(),
    'total_epochs': pd.Series(),
    'time_per_epoch': pd.Series(),
    'dataset_size': pd.Series(),
    'width': pd.Series(),
    'height': pd.Series(),
    'depth': pd.Series()
}

for setup in setups:
    checkpoint = f'{setup["arch"]}_{setup["loss_fn"]}_f{setup["filters"]}_bs{setup["batch_size"]}_bn{setup["batch_norm"]}'

    # Grab dataset
    my_dataset = dataset.MyDataset(
        collection_name = cfg.dataset['collection_name'],
        batch_size=setup['batch_size']
    )

    # Create model
    my_model = model.MyModel(
        arch = setup['arch'],
        loss_function = setup['loss_fn'],
        batch_norm = setup['batch_norm'],
        filters = setup['filters'],
        threshold=cfg.model['threshold'],
        checkpoint = checkpoint,
        width = cfg.dataset['width'],
        height = cfg.dataset['height'],
        depth = cfg.dataset['depth'],
        epochs = cfg.model['epochs']
    )

    # Create generators
    train_generator, test_generator = my_dataset.create_train_test_gen()
    my_model.get_model_summary()

    # Train model
    history, epoch_time = my_model.train(train_generator, test_generator)
    epochs = len(history.history['val_loss'])

    # Validate model
    val_loss, val_acc = my_model.evaluate(test_generator)
    ds_size = my_dataset.get_count()
    
    # Save model parameters and performance
    with open(f'output/models/{checkpoint}.setup.txt', 'w') as text_file:
        print(f'Architecture: {setup["arch"]}', file=text_file)
        print(f'Loss fn: {setup["loss_fn"]}', file=text_file)
        print(f'Filters: {setup["filters"]}', file=text_file)
        print(f'Batch size: {setup["batch_size"]}', file=text_file)
        print(f'Batch normalization: {setup["batch_norm"]}', file=text_file)
        print('---', file=text_file)
        print(f'Validation loss: {val_loss}', file=text_file)
        print(f'Validation accuracy: {val_acc}', file=text_file)
        print(f'Total epochs: {epochs}', file=text_file)
        print(f'Time per epoch: {epoch_time}', file=text_file)
        print('---', file=text_file)
        print(f'Checkpoint: {checkpoint}', file=text_file)
        print(f'Dataset size: {ds_size}', file=text_file)
        print(f'Width: {cfg.dataset["width"]}', file=text_file)
        print(f'Height: {cfg.dataset["height"]}', file=text_file)
        print(f'Depth: {cfg.dataset["depth"]}', file=text_file)

    # save csv
    params['arch'][checkpoint] = setup['arch']
    params['loss_fn'][checkpoint] = setup['loss_fn']
    params['batch_size'][checkpoint] = setup['batch_size']
    params['filters'][checkpoint] = setup['filters']
    params['batch_norm'][checkpoint] = setup['batch_norm']
    params['val_loss'][checkpoint] = val_loss
    params['val_acc'][checkpoint] = val_acc
    params['total_epochs'][checkpoint] = epochs
    params['time_per_epoch'][checkpoint] = epoch_time
    params['dataset_size'][checkpoint] = ds_size
    params['width'][checkpoint] = cfg.dataset['width']
    params['height'][checkpoint] = cfg.dataset['height']
    params['depth'][checkpoint] = cfg.dataset['depth']

    output = pd.DataFrame(params)
    output.to_csv(f'output/models/{cfg.dataset["collection_name"]}_summary.csv')
    
    X_preds, y_preds = my_model.predict(test_generator)
    X_test, y_test = test_generator.__getitem__(0)
    
    image = X_test[0].squeeze()
    image_0 = image[25, :, :]
    image_1 = image[:, 20, :]
    image_2 = image[:, :, 25]
    
    mask = y_test[0].squeeze()
    mask_0 = mask[25, :, :]
    mask_1 = mask[:, 20, :]
    mask_2 = mask[:, :, 25]
    
    pred = y_preds[0].squeeze()
    pred_0 = pred[25, :, :]
    pred_1 = pred[:, 20, :]
    pred_2 = pred[:, :, 25]
    
    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    
    ax[0][0].imshow(image_0, cmap='gray')
    ax[0][1].imshow(image_1, cmap='gray')
    ax[0][2].imshow(image_2, cmap='gray')
    
    ax[1][0].imshow(mask_0, cmap='gray')
    ax[1][1].imshow(mask_1, cmap='gray')
    ax[1][2].imshow(mask_2, cmap='gray')
    
    ax[2][0].imshow(pred_0, cmap='gray')
    ax[2][1].imshow(pred_1, cmap='gray')
    ax[2][2].imshow(pred_2, cmap='gray')
    
    fig.savefig(f'output/models/{checkpoint}.png')
