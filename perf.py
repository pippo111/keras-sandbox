import pandas as pd

from common import model
from common import dataset
from common import plots
from common.utils import get_all_gen_items, calc_confusion_matrix
from common.logs import to_table
import config as cfg

from train import train

params = {
    'arch': pd.Series(),
    'loss_fn': pd.Series(),
    'optimizer_fn': pd.Series(),
    'batch_size': pd.Series(),
    'filters': pd.Series(),
    'batch_norm': pd.Series(),
    'val_loss': pd.Series(),
    'val_acc': pd.Series(),
    'fp_rate': pd.Series(),
    'fn_rate': pd.Series(),
    'fp_total': pd.Series(),
    'fn_total': pd.Series(),
    'total_epochs': pd.Series(),
    'time_per_epoch': pd.Series(),
    'dataset_size': pd.Series(),
    'width': pd.Series(),
    'height': pd.Series(),
    'depth': pd.Series(),
    'slice_depth': pd.Series()
}

for setup in cfg.setups:
    # Train model
    checkpoint, epoch_total, epoch_time = train({ **cfg.dataset, **cfg.model, **setup })

    # # Validate model
    # my_model.load()
    # val_loss, val_acc = my_model.evaluate(test_generator)
    # ds_size = my_dataset.get_count()
    
    # X_preds, y_preds = my_model.predict(test_generator)
    # X_test, y_test = get_all_gen_items(test_generator)

    # # calculate false and true positive and negative
    # fp_rate, fn_rate, fp_total, fn_total = calc_confusion_matrix(y_test, y_preds)

    # # save plot with sample image and result mask
    # image = X_test[0].squeeze()
    # mask = y_test[0].squeeze()
    # pred = y_preds[0].squeeze()

    # plots.save_sample_plot(
    #     image,
    #     mask,
    #     pred,
    #     filename=f'output/models/{checkpoint}.png',
    #     patch=(cfg.logs['axis_0'], cfg.logs['axis_1'], cfg.logs['axis_2']))

    # params['arch'][checkpoint] = setup['arch']
    # params['loss_fn'][checkpoint] = setup['loss_fn']
    # params['optimizer_fn'][checkpoint] = setup['optimizer_fn']
    # params['batch_size'][checkpoint] = setup['batch_size']
    # params['filters'][checkpoint] = setup['filters']
    # params['batch_norm'][checkpoint] = setup['batch_norm']
    # params['val_loss'][checkpoint] = val_loss
    # params['val_acc'][checkpoint] = val_acc
    # params['fp_rate'][checkpoint] = fp_rate
    # params['fn_rate'][checkpoint] = fn_rate
    # params['fp_total'][checkpoint] = fp_total
    # params['fn_total'][checkpoint] = fn_total
    # params['total_epochs'][checkpoint] = epoch_total
    # params['time_per_epoch'][checkpoint] = epoch_time
    # params['dataset_size'][checkpoint] = ds_size
    # params['width'][checkpoint] = cfg.dataset['width']
    # params['height'][checkpoint] = cfg.dataset['height']
    # params['depth'][checkpoint] = cfg.dataset['depth']
    # params['slice_depth'][checkpoint] = cfg.dataset['slice_depth']

    # output = pd.DataFrame(params)

    # # Save as csv
    # output.to_csv(f'output/models/{cfg.dataset["collection_name"]}_summary.csv')

    # # Save as html interactive table
    # to_table(output.to_html(), f'output/models/{cfg.dataset["collection_name"]}_summary.html')
