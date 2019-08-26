import pandas as pd

from common import model
from common import dataset
from common import plots
from common.utils import get_all_gen_items, calc_confusion_matrix
from common.logs import to_table
import config as cfg

from train import train
from evaluate import evaluate
from plot import plot

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
    'input_shape': pd.Series()
}

for setup in cfg.setups:
    # Train model
    results = train({ **cfg.dataset, **cfg.model, **setup })
    
    checkpoint = results['checkpoint']

    # Evaluate model
    evaluation = evaluate(
        checkpoint,
        cfg.dataset['collection_name'],
        setup['batch_size'],
        cfg.dataset['limit'],
        cfg.model['threshold']
    )

    print(evaluation)

    plot(
        checkpoint,
        cfg.dataset['collection_name'],
        coords = (cfg.logs['axis_0'], cfg.logs['axis_1'], cfg.logs['axis_2']),
        batch_size = setup['batch_size'],
        limit = cfg.dataset['limit'],
        threshold = cfg.model['threshold']
    )

    params['arch'][checkpoint] = setup['arch']
    params['loss_fn'][checkpoint] = setup['loss_fn']
    params['optimizer_fn'][checkpoint] = setup['optimizer_fn']
    params['batch_size'][checkpoint] = setup['batch_size']
    params['filters'][checkpoint] = setup['filters']
    params['batch_norm'][checkpoint] = setup['batch_norm']
    params['val_loss'][checkpoint] = evaluation['val_loss']
    params['val_acc'][checkpoint] = evaluation['val_acc']
    params['fp_rate'][checkpoint] = evaluation['fp_rate']
    params['fn_rate'][checkpoint] = evaluation['fn_rate']
    params['fp_total'][checkpoint] = evaluation['fp_total']
    params['fn_total'][checkpoint] = evaluation['fn_total']
    params['total_epochs'][checkpoint] = results['epoch_total']
    params['time_per_epoch'][checkpoint] = results['epoch_time']
    params['dataset_size'][checkpoint] = evaluation['ds_size']
    params['input_shape'][checkpoint] = cfg.dataset['input_shape']

    output = pd.DataFrame(params)

    # Save as csv
    output.to_csv(f'output/models/{cfg.dataset["collection_name"]}_summary.csv')

    # Save as html interactive table
    to_table(output.to_html(), f'output/models/{cfg.dataset["collection_name"]}_summary.html')
