import config as cfg

from common.model import MyModel
from common.dataset import MyDataset
from common.utils import calc_weights_generator

for setup in cfg.setups:
    # Grab dataset
    my_dataset = MyDataset(
        collection_name = cfg.dataset['collection_name'],
        batch_size = setup['batch_size'],
        limit = cfg.dataset['limit']
    )

    # Create generators
    train_generator, valid_generator, test_generator = my_dataset.get_generators()

    # Only for weighted loss functions
    if setup['loss_fn'] == 'wce':
        loss_weights = calc_weights_generator(train_generator)
    else:
        loss_weights = None

    # Create model
    my_model = MyModel(
            train_generator = train_generator,
            valid_generator = valid_generator,
            test_generator = test_generator,
            struct = cfg.model['struct'],
            arch = setup['arch'],
            loss_fn = setup['loss_fn'],
            optimizer_fn = setup['optimizer_fn'],
            batch_size = setup['batch_size'],
            batch_norm = setup['batch_norm'],
            filters = setup['filters'],
            input_shape = cfg.dataset['input_shape'],
            threshold = setup['threshold']
        )

    my_model.create(load_weights=False, loss_weights=loss_weights)
    my_model.print_summary()

    # Train model
    my_model.train(epochs = cfg.model['epochs'])

    # Evaluate model
    my_model.evaluate()

    # Plot sample output result
    # my_model.plot_result(
    #     coords = (cfg.logs['axis_0'], cfg.logs['axis_1'], cfg.logs['axis_2']),
    #     show = False, save = True
    # )

    # Save results
    my_model.save_results(f'output/models/{cfg.dataset["collection_name"]}_results')
