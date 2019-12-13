import argparse

import torch

from data_loader import DataLoader
from helpers import create_csv_submission, make_prediction
from model import MultiLayerGRU

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str,
                        help='Path to test file (.txt)',
                        required=True)
    parser.add_argument('-o', '--out', type=str,
                        help='Path to submission file (.csv)',
                        required=True)                    
    parser.add_argument('-m', '--model', type=str,
                        help='Path to model file (.pth)',
                        required=True)
    parser.add_argument('-v', '--vocabulary', type=str,
                        help='Path to vocabulary file (.pth)',
                        required=True)
    args = parser.parse_args()

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Loading the vocabulary and the model')
    saved_model = torch.load(
        args.model, map_location=lambda storage, loc: storage)
    embedding_dim = saved_model['embedding_dim']
    data = DataLoader(text_field_file=args.vocabulary,
                      embedding_dim=embedding_dim)
    model = MultiLayerGRU(embedding_dim, data.get_vector(), device)
    model.load_state_dict(saved_model['model_state'])
    model.to(device)

    test_batch_it = data.load_test(args.file, device)
    print('Making the prediction')
    ids, pred = make_prediction(model, test_batch_it)

    print('Outputing submission')
    create_csv_submission(ids, pred, args.out)
