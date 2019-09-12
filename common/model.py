from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import os

from common.utils import get_all_gen_items, calc_confusion_matrix, calc_precision, calc_recall, calc_f1score
from common.plots import plot_confusions
from common.logs import to_table
from common.time_callback import TimeHistory
from common.reinit_falsestart_callback import ReinitWeightOnFalseStart
from networks import network
from networks import loss
from networks import optimizer

class MyModel():
    def __init__(
            self,
            train_generator, valid_generator, test_generator,
            struct, arch, loss_fn, optimizer_fn,
            batch_size=16, batch_norm=False, filters=16,
            threshold=0.5, input_shape=(48,64,64)
        ):
        self.train_generator = train_generator
        self.valid_generator = valid_generator
        self.test_generator = test_generator

        self.checkpoint = "{}_{}_{}_{}_bs-{}_bn-{}_f-{}".format(
                struct,
                arch,
                optimizer_fn,
                loss_fn,
                batch_size,
                batch_norm,
                filters
            )
        
        self.setup = {
            'arch': arch,
            'loss_fn': loss_fn,
            'optimizer_fn': optimizer_fn,
            'batch_size': batch_size,
            'filters': filters,
            'batch_norm': batch_norm,
            'train_sets': train_generator.__len__() * batch_size,
            'input_shape': input_shape,
            'threshold': threshold
        }

        self.results = {
            'val_loss': '',
            'val_acc': '',
            'fp_rate': '',
            'fn_rate': '',
            'fp_total': '',
            'fn_total': '',
            'f_total': '',
            'precision': '',
            'recall': '',
            'f1score': '',
            'total_epochs': '',
            'time_per_epoch': ''
        }

    """Creates and compile model with given hyperparameters
    It is possible to load model weights from previous training
    """
    def create(self, load_weights=False, loss_weights=None, filename=''):
        # if we want to use 2d model arch we need to squeeze tuple
        squeezed_input_shape = tuple(x for x in self.setup['input_shape'] if x > 1)

        self.model = network.get(
                name = self.setup['arch'],
                loss_function = loss.get(self.setup['loss_fn'], loss_weights),
                optimizer_function = optimizer.get(self.setup['optimizer_fn']),
                batch_norm = self.setup['batch_norm'],
                input_shape = squeezed_input_shape,
                n_filters = self.setup['filters']
            )

        filename = filename or self.checkpoint

        if load_weights:
            self.model.load_weights(f'output/models/{filename}.hdf5')


    """Perform model training
    Returns basic info about trainig time
    """
    def train(self, epochs=50):
        time_callback = TimeHistory()

        if self.setup['optimizer_fn'] == 'RAdam':
            callbacks = [
                time_callback,
                ReinitWeightOnFalseStart(patience=3, trials=1, checks=10, verbose=1),
                ModelCheckpoint(f'output/models/{self.checkpoint}.hdf5', verbose=1, save_best_only=True)
            ]
        else:
            callbacks = [
                time_callback,
                ReinitWeightOnFalseStart(patience=3, trials=1, checks=10, verbose=1),
                ReduceLROnPlateau(factor=0.1, patience=6, min_lr=0.00001, verbose=1),
                ModelCheckpoint(f'output/models/{self.checkpoint}.hdf5', verbose=1, save_best_only=True)
            ]

        history = self.model.fit_generator(
            self.train_generator,
            epochs = epochs,
            callbacks = callbacks,
            validation_data = self.valid_generator
        )

        times = time_callback.times
        epoch_time = int(np.mean(times))
        epoch_total = len(history.history['val_loss'])

        self.results['total_epochs'] = epoch_total
        self.results['time_per_epoch'] = epoch_time

        return history, epoch_time
    
    """Evaluates model performance
    by calculating confusion matrix

    Returns calculated values
    """
    def evaluate(self):
        # Validate model
        val_loss, val_acc = self.model.evaluate_generator(self.valid_generator, verbose=1)
        
        y_preds = self.predict(self.valid_generator, self.setup['threshold'])
        dummy_X_test, y_test = get_all_gen_items(self.valid_generator)

        # Calculate false and true positive and negative
        fp_rate, fn_rate, fp_total, fn_total, f_total = calc_confusion_matrix(y_test, y_preds)
        precision = calc_precision(y_test, y_preds)
        recall = calc_recall(y_test, y_preds)
        f1score = calc_f1score(precision, recall)

        self.results['val_loss'] = val_loss
        self.results['val_acc'] = val_acc
        self.results['fp_rate'] = fp_rate
        self.results['fn_rate'] = fn_rate
        self.results['fp_total'] = fp_total
        self.results['fn_total'] = fn_total
        self.results['f_total'] = f_total
        self.results['precision'] = precision
        self.results['recall'] = recall
        self.results['f1score'] = f1score
        
        return fp_rate, fn_rate, fp_total, fn_total, f_total

    """Returns predicted region and segmented representation
    """
    def predict(self, generator, threshold):
        preds = self.model.predict_generator(generator, verbose=1)
        preds = (preds > threshold).astype(np.uint8)
        
        return preds

    """Display a plot with sample image and prediction
    It is possible to only save image as png
    """
    def plot_result(self, coords, show=True, save=False):
        y_preds = self.predict(self.test_generator, self.setup['threshold'])
        X_test, y_test = get_all_gen_items(self.test_generator)

        image = X_test[15].squeeze()
        mask = y_test[15].squeeze()
        pred = y_preds[15].squeeze()

        plot_confusions(
            image,
            mask,
            pred,
            filename = f'output/models/{self.checkpoint}.png',
            coords = coords,
            show = show,
            save = save
        )

    """Returns dictionary with setup
    """
    def save_results(self, filename):
        csv_file = f'{filename}.csv'
        html_file = f'{filename}.html'

        results = [{ 'checkpoint': self.checkpoint, **self.setup, **self.results }]
        output = pd.DataFrame(results)

        if not os.path.exists(csv_file):
            output.to_csv(csv_file, index=False, header=True, mode='a')
        else:
            output.to_csv(csv_file, index=False, header=False, mode='a')

        generated_csv = pd.read_csv(csv_file)

        #open csv
        to_table(generated_csv.to_html(index=False), html_file)
            

    """Prints out model architecture to standard output
    """
    def print_summary(self):
        self.model.summary()