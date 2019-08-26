from time import time
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import load_model
import numpy as np

from common.time_callback import TimeHistory
from common.reinit_falsestart_callback import ReinitWeightOnFalseStart
from networks import network
from networks import loss
from networks import optimizer
import config as cfg

class MyModel():
    def __init__(self, checkpoint, epochs=50, threshold=0.5):
        self.checkpoint = checkpoint
        self.epochs = epochs
        self.threshold = threshold

    # from_arch
    def create(
            self,
            arch,
            loss_function,
            optimizer_function,
            batch_norm=False,
            filters=16,
            input_shape=(48,64,64)
        ):
        self.model = network.get(
                name=arch,
                loss_function=loss.get(loss_function),
                optimizer_function=optimizer.get(optimizer_function),
                batch_norm=batch_norm,
                input_shape=input_shape,
                n_filters=filters
            )
        
    # from_checkpoint
    def load(self):
        self.model = load_model(f'output/models/{self.checkpoint}.hdf5')

    def train(self, train_generator, test_generator):
        time_callback = TimeHistory()

        history = self.model.fit_generator(
            train_generator,
            epochs=self.epochs,
            callbacks=[
                time_callback,
                ReinitWeightOnFalseStart(patience=3, trials=1, checks=10, verbose=1),
                # TensorBoard(log_dir=f'output/logs/{time()}-{self.checkpoint}'),
                # EarlyStopping(patience=10, verbose=1),
                # ReduceLROnPlateau(factor=0.1, patience=6, min_lr=0.00001, verbose=1),
                ModelCheckpoint(f'output/models/{self.checkpoint}.hdf5', verbose=1, save_best_only=True)
            ],
            validation_data=test_generator
        )

        times = time_callback.times
        epoch_time = int(np.mean(times))

        return history, epoch_time
        
    def evaluate(self, test_generator):
        history = self.model.evaluate_generator(test_generator, verbose=1)
        
        return history

    def predict(self, test_generator):
        X_preds = self.model.predict_generator(test_generator, verbose=1)
        y_preds = (X_preds > self.threshold).astype(np.uint8)
        
        return X_preds, y_preds

    def print_summary(self):
        self.model.summary()