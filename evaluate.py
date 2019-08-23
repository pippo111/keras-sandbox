from common.model import MyModel
from common.dataset import MyDataset
from common.utils import get_all_gen_items, calc_confusion_matrix

def evaluate(checkpoint, collection_name, batch_size=32, limit=None, threshold=0.5):
    # Grab dataset
    my_dataset = MyDataset(
        collection_name = collection_name,
        batch_size = batch_size,
        limit = limit
    )

    # Load model
    my_model = MyModel(checkpoint = checkpoint, threshold = threshold)
    my_model.load()
    my_model.print_summary()

    # Create generators
    _, test_generator = my_dataset.create_train_test_gen()

    # Validate model
    val_loss, val_acc = my_model.evaluate(test_generator)
    ds_size = my_dataset.get_count()
    
    _, y_preds = my_model.predict(test_generator)
    _, y_test = get_all_gen_items(test_generator)

    # Calculate false and true positive and negative
    fp_rate, fn_rate, fp_total, fn_total = calc_confusion_matrix(y_test, y_preds)

    evaluation = {
        'val_loss': val_loss,
        'val_acc': val_acc,
        'fp_rate': fp_rate,
        'fn_rate': fn_rate,
        'fp_total': fp_total,
        'fn_total': fn_total,
        'ds_size': ds_size
    }

    return evaluation