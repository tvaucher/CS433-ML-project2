import csv
import random

import torch


class BatchGenerator:
    '''Wrapper around BucketIterator, generate for every batch a tuple (features, target)'''

    def __init__(self, dl, x_field, y_field):
        self.dl, self.x_field, self.y_field = dl, x_field, y_field

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield (X, y)


def set_seed(seed=1):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_prediction(model, test_batch_it):
    test_pred = []
    test_ids = []
    model.train(False)
    for (seq, lengths), ids in test_batch_it:
        pred = model(seq, lengths).argmax(axis=1)
        pred[pred == 0] = -1
        test_pred.append(pred)
        test_ids.append(ids)

    return torch.cat(test_ids).cpu().data.numpy(), torch.cat(test_pred).cpu().data.numpy()


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})
