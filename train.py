import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange

from helpers import set_seed
from data_loader import DataLoader
from model import BaselineGRU, load_model, save_model


def compute_accuracy(pred, targets):
    return (pred == targets).float().mean().item()


def log_validation(model, loss_fn, val_batch_it):
    val_pred = []
    val_target = []
    val_loss = 0.
    for (seq, lengths), targets in val_batch_it:
        pred = model(seq, lengths)
        loss = loss_fn(pred, targets)
        val_pred.append(pred.argmax(axis=1))
        val_target.append(targets)
        val_loss += loss.item()
    print('Validation', compute_accuracy(
        torch.cat(val_pred), torch.cat(val_target)), val_loss)


def fit(model, loss_fn, optimizer, train_batch_it, val_batch_it, checkpoint_name, embedding_dim, epochs=10, nb_epochs_done=0):
    for epoch in trange(nb_epochs_done, epochs):
        model.train(True)
        train_pred = []
        train_targets = []
        train_loss = 0.

        batch = tqdm(train_batch_it, leave=False,
                     total=len(train_batch_it), desc=f'Epoch {epoch}')
        for (seq, lengths), targets in batch:
            pred = model(seq, lengths)
            loss = loss_fn(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_pred.append(pred.argmax(axis=1))
            train_targets.append(targets)
            train_loss += loss.item()

        print(epoch, compute_accuracy(torch.cat(train_pred),
                                      torch.cat(train_targets)), train_loss)

        save_model(checkpoint_name, model, optimizer, epoch, embedding_dim)

        model.train(False)
        log_validation(model, loss_fn, val_batch_it)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str,
                        help='Path to preprocessed training file (.tsv)',
                        required=True)
    parser.add_argument('-g', '--glove', type=str,
                        help='Path to the folder containing the pretrained GloVe',
                        required=True)
    parser.add_argument('-c', '--checkpoint', type=str,
                        help='Path to checkpoint prefix',
                        required=True)
    parser.add_argument('-n', '--epochs', type=int, default=10,
                        help='Number of epoch to train for default=10')
    parser.add_argument('--seed', type=int, default=1,
                        help='Set the seed, default=1')
    parser.add_argument('--dim', type=int, default=200, choices=[
                        25, 50, 100, 200], help='Set the dimension of the embedding [25, 50, 100, 200], default=200')
    args = parser.parse_args()
    set_seed(args.seed)

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Loading the train data')
    data = DataLoader(embedding_dim=args.dim)
    train_batch_it, val_batch_it = data.load_split_train(
        args.file, args.glove, device)

    print('Saving the vocabulary')
    data.save()

    print('Creating the model')
    model = BaselineGRU(args.dim, data.get_vector(), device).to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 1e-3)
    loss_criterion = nn.NLLLoss()
    nb_epochs_done = load_model(args.checkpoint, model, optimizer)

    print('Fitting the model')
    fit(model, loss_criterion, optimizer, train_batch_it, val_batch_it, args.checkpoint,\
        args.dim, epochs=args.epochs, nb_epochs_done=nb_epochs_done)
