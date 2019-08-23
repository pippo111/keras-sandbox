from common.model import MyModel
from common.dataset import MyDataset

def train(setup):
    # Create filename for model checkpoint
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
        batch_size = setup['batch_size'],
        limit = setup['limit']
    )

    # Create model
    my_model = MyModel(checkpoint = checkpoint, epochs = setup['epochs'])
    my_model.create(
        arch = setup['arch'],
        loss_function = setup['loss_fn'],
        optimizer_function = setup['optimizer_fn'],
        batch_norm = setup['batch_norm'],
        filters = setup['filters'],
        width = setup['width'],
        height = setup['height'],
        depth = setup['depth'],
        slice_depth = setup['slice_depth']
    )
    my_model.print_summary()


    # Create generators
    train_generator, valid_generator, test_generator = my_dataset.create_train_valid_test_gen()

    # Perform train
    history, epoch_time = my_model.train(train_generator, valid_generator)
    epoch_total = len(history.history['val_loss'])

    results = {
        'checkpoint': checkpoint,
        'epoch_total': epoch_total,
        'epoch_time': epoch_time
    }

    return results
