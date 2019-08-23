from common.model import MyModel
from common.dataset import MyDataset
from common.utils import get_all_gen_items, calc_confusion_matrix
from common.plots import plot_confusions

def plot(checkpoint, collection_name, coords, batch_size=32, limit=None, threshold=0.5):
    my_dataset = MyDataset(
        collection_name = collection_name,
        batch_size = batch_size,
        limit = limit
    )
    train_generator, valid_generator, test_generator = my_dataset.create_train_valid_test_gen()

    my_model = MyModel(checkpoint = checkpoint, threshold = threshold)
    my_model.load()
    
    X_preds, y_preds = my_model.predict(test_generator)
    X_test, y_test = test_generator.__getitem__(0)

    # save plot with sample image and result mask
    image = X_test[0].squeeze()
    mask = y_test[0].squeeze()
    pred = y_preds[0].squeeze()

    plot_confusions(
        image,
        mask,
        pred,
        filename = f'output/models/{checkpoint}.png',
        coords = coords,
        show = False,
        save = True
    )
