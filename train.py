from common.model import MyModel
from common.dataset import MyDataset

def train(setup):
    checkpoint = "{}_{}_{}_bs-{}_bn-{}_f-{}".format(
        setup['arch'],
        setup['optimizer_fn'],
        setup['loss_fn'],
        setup['batch_size'],
        setup['batch_norm'],
        setup['filters']
    )

    # Grab dataset
    my_dataset = MyDataset(
        collection_name = setup['collection_name'],
        batch_size=setup['batch_size'],
        limit = setup['limit']
    )

    # Create model
    my_model = MyModel(
        arch = setup['arch'],
        loss_function = setup['loss_fn'],
        optimizer_function = setup['optimizer_fn'],
        batch_norm = setup['batch_norm'],
        filters = setup['filters'],
        threshold=setup['threshold'],
        checkpoint = checkpoint,
        width = setup['width'],
        height = setup['height'],
        depth = setup['depth'],
        slice_depth = setup['slice_depth'],
        epochs = setup['epochs']
    )

    # Create generators
    train_generator, test_generator = my_dataset.create_train_test_gen()
    my_model.get_model_summary()

    history, epoch_time = my_model.train(train_generator, test_generator)
    epoch_total = len(history.history['val_loss'])

    return checkpoint, epoch_total, epoch_time
