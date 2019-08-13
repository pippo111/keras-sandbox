import matplotlib
import matplotlib.pyplot as plt

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
    'arch': 'ResUnet3d', 'loss_fn': 'binary',
    'batch_size': 4, 'filters': 8, 'batch_norm': False
  },
  {
    'arch': 'ResUnet3d', 'loss_fn': 'binary',
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
    'arch': 'ResUnet3d', 'loss_fn': 'binary',
    'batch_size': 1, 'filters': 16, 'batch_norm': False
  },
  {
    'arch': 'ResUnet3d', 'loss_fn': 'binary',
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
    'arch': 'ResUnet3d', 'loss_fn': 'binary',
    'batch_size': 8, 'filters': 16, 'batch_norm': False
  },
  {
    'arch': 'ResUnet3d', 'loss_fn': 'binary',
    'batch_size': 8, 'filters': 16, 'batch_norm': True
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
    'arch': 'ResUnet3d', 'loss_fn': 'dice',
    'batch_size': 4, 'filters': 8, 'batch_norm': False
  },
  {
    'arch': 'ResUnet3d', 'loss_fn': 'dice',
    'batch_size': 4, 'filters': 8, 'batch_norm': True
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
    'arch': 'ResUnet3d', 'loss_fn': 'dice',
    'batch_size': 1, 'filters': 16, 'batch_norm': False
  },
  {
    'arch': 'ResUnet3d', 'loss_fn': 'dice',
    'batch_size': 1, 'filters': 16, 'batch_norm': True
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
    'arch': 'ResUnet3d', 'loss_fn': 'dice',
    'batch_size': 8, 'filters': 16, 'batch_norm': False
  },
  {
    'arch': 'ResUnet3d', 'loss_fn': 'dice',
    'batch_size': 8, 'filters': 16, 'batch_norm': True
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
    'arch': 'Unet3d', 'loss_fn': 'binary',
    'batch_size': 4, 'filters': 8, 'batch_norm': False
  },
  {
    'arch': 'Unet3d', 'loss_fn': 'binary',
    'batch_size': 4, 'filters': 8, 'batch_norm': True
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
    'arch': 'Unet3d', 'loss_fn': 'binary',
    'batch_size': 1, 'filters': 16, 'batch_norm': False
  },
  {
    'arch': 'Unet3d', 'loss_fn': 'binary',
    'batch_size': 1, 'filters': 16, 'batch_norm': True
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
    'arch': 'Unet3d', 'loss_fn': 'binary',
    'batch_size': 8, 'filters': 16, 'batch_norm': False
  },
  {
    'arch': 'Unet3d', 'loss_fn': 'binary',
    'batch_size': 8, 'filters': 16, 'batch_norm': True
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
    'arch': 'Unet3d', 'loss_fn': 'dice',
    'batch_size': 4, 'filters': 8, 'batch_norm': False
  },
  {
    'arch': 'Unet3d', 'loss_fn': 'dice',
    'batch_size': 4, 'filters': 8, 'batch_norm': True
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
    'arch': 'Unet3d', 'loss_fn': 'dice',
    'batch_size': 1, 'filters': 16, 'batch_norm': False
  },
  {
    'arch': 'Unet3d', 'loss_fn': 'dice',
    'batch_size': 1, 'filters': 16, 'batch_norm': True
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
    'arch': 'Unet3d', 'loss_fn': 'dice',
    'batch_size': 8, 'filters': 16, 'batch_norm': False
  },
  {
    'arch': 'Unet3d', 'loss_fn': 'dice',
    'batch_size': 8, 'filters': 16, 'batch_norm': True
  }
]

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
  
  # Save model parameters and performance
  with open(f'output/models/{cfg.model["checkpoint"]}.setup.txt', 'w') as text_file:
    print(f'Architecture: {cfg.model["arch"]}', file=text_file)
    print(f'Loss fn: {cfg.model["loss_fn"]}', file=text_file)
    print(f'Filters: {cfg.model["filters"]}', file=text_file)
    print(f'Batch size: {cfg.model["batch_size"]}', file=text_file)
    print(f'Batch normalization: {cfg.model["batch_norm"]}', file=text_file)
    print('---', file=text_file)
    print(f'Validation loss: {val_loss}', file=text_file)
    print(f'Validation accuracy: {val_acc}', file=text_file)
    print(f'Total epochs: {epochs}', file=text_file)
    print(f'Time per epoch: {epoch_time}', file=text_file)
    print('---', file=text_file)
    print(f'Checkpoint: {cfg.model["checkpoint"]}', file=text_file)
    print(f'Width: {cfg.dataset["width"]}', file=text_file)
    print(f'Height: {cfg.dataset["height"]}', file=text_file)
    print(f'Depth: {cfg.dataset["depth"]}', file=text_file)

  
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