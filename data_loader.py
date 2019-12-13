import random
import torch
import torchtext
from torchtext.data import (BucketIterator, Dataset, Example, Field, Iterator,
                            TabularDataset)
from torchtext.vocab import Vectors

from helpers import BatchGenerator
from preprocessing import filter_tokens, tokenize


class DataLoader:
    ''' 
    Housekeeping module for the train and test data.
    Load and transform it into batches that can fed to the model
    '''
    def __init__(self, text_field_file=None, embedding_dim=200):
        ''' text_field_file: vocabulary file, should be not None for run.py '''
        self.label_field = Field(
            sequential=False, use_vocab=False, is_target=True)
        self.text_field = torch.load(text_field_file) if text_field_file else Field(
            include_lengths=True, preprocessing=filter_tokens)
        self.id_field = Field(sequential=False, use_vocab=False)

        self.__train_fields = [('label', self.label_field),
                               ('tweet', self.text_field)]
        self.train_batch_sizes = (512, 1024)
        self.__test_fields = [('id', self.id_field),
                              ('tweet', self.text_field)]
        self.test_batch_size = 1000

        self.embedding_dim = embedding_dim

    def save(self, path='model/'):
        ''' Save the vocabulary, used in train.py '''
        torch.save(self.text_field, path + 'vocabulary.pth')

    def get_vector(self):
        return self.text_field.vocab.vectors

    def load_split_train(self, path_train, path_glove, device, split_ratio=0.9):
        ''' Load, split and transform the training set into train / val batches '''
        # Load Dataset
        dataset = TabularDataset(
            path_train, 'tsv', self.__train_fields, skip_header=True)

        # Split Dataset and create corresponding vocabulary
        train, val = dataset.split(
            split_ratio=split_ratio, random_state=random.getstate())
        vectors = Vectors(
            f'glove.twitter.27B.{self.embedding_dim}d.txt', cache=path_glove)
        self.text_field.build_vocab(train, val, vectors=vectors)

        # Create the Iterators (similar to torch.DataLoader) for the text
        train_batch, val_batch = BucketIterator.splits(
            (train, val),
            batch_sizes=self.train_batch_sizes,
            device=device,
            # Need to sort them according to length
            sort_key=lambda x: len(x.tweet),
            sort_within_batch=True,
            repeat=False
        )

        train_batch_it = BatchGenerator(train_batch, 'tweet', 'label')
        val_batch_it = BatchGenerator(val_batch, 'tweet', 'label')
        return train_batch_it, val_batch_it

    def load_test(self, path_test, device):
        ''' Load, split and transform the test set into batches '''
        with open(path_test, 'r', encoding='utf-8') as test_file:
            test_lines = [line.rstrip('\n').split(',', 1)
                          for line in test_file]
            test_examples = list(map(lambda x: Example.fromlist(
                [x[0], ' '.join(tokenize(x[1]))], self.__test_fields), test_lines))

        test_dataset = Dataset(test_examples, self.__test_fields)
        test_batch = Iterator(test_dataset, self.test_batch_size, sort_key=lambda x: len(x.tweet), device=device, sort_within_batch=True, train=False)
        test_batch_it = BatchGenerator(test_batch, 'tweet', 'id')
        return test_batch_it
