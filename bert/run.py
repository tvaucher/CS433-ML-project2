import argparse
import torch
import pandas as pd
from helpers import BatchGenerator, create_csv_submission, get_device
from transformers import BertForSequenceClassification
from tqdm.auto import tqdm

class TestModel:
    '''
    Model for producing the prediction on the test set
    Load the model and the test set before producing the output
    '''
    def __init__(self, model_fn, model_path, test_path, batch_size, device):
        self.device = device
        self.model = model_fn.from_pretrained(model_path).to(device)
        test = pd.read_pickle(test_path)
        test.label = test.label.apply(int)
        self.test_batch = BatchGenerator(test, batch_size, device, shuffle=False)
    
    def make_predictions(self):
        ''' Make predicitons for test set and return them '''
        test_pred = []
        test_ids = []
        self.model.eval()
        for seq, mask, labels in tqdm(self.test_batch):
            pred = self.model(seq, attention_mask=mask)[0].argmax(axis=1)
            pred[pred == 0] = -1
            test_pred.append(pred)
            test_ids.append(labels)
        return torch.cat(test_ids).cpu().data.numpy(),\
               torch.cat(test_pred).cpu().data.numpy()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str,
                        help='Path to test encoded file (.pkl.gz)',
                        required=True)
    parser.add_argument('-o', '--out', type=str,
                        help='Path to submission file (.csv)',
                        required=True)                    
    parser.add_argument('-m', '--model', type=str,
                        help='Path to model folder',
                        required=True)
    parser.add_argument('--batch_size', type=int,
                        help='Batch size (2GB GPU==10, else 25), default=10',
                        default=10)
    args = parser.parse_args()
    
    print('Loading the model and the test set')
    model = TestModel(BertForSequenceClassification, args.model, args.file, args.batch_size, get_device())
    print('Infering the prediction')
    test_ids, test_pred = model.make_predictions()
    print(f'Creating the submission under {args.out}')
    create_csv_submission(test_ids, test_pred, args.out)