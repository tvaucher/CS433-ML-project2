import csv
import random

import torch
from torch.nn.utils.rnn import pad_sequence

import numpy as np

def set_seed(seed):
    ''' Set all possible seed to given seed'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    ''' Check if CUDA is available '''
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

class BatchGenerator:
    '''
    Helper class to transform data into iterable batches
    Put all the data on the GPU so it doesn't have to move it around (tested on K80/T4/P100)
    Can be shuffled between epoch
    '''
    def __init__(self, df, batch_size, device, shuffle=True):
        self.shuffle = shuffle
        batches = list(range(0, len(df), batch_size))
        self.__len = len(batches)
        self.__batch_id = list(range(self.__len))
        
        self.__labels = torch.tensor(df.label.to_list()).to(device)
        self.__seq = pad_sequence(df.tensor.to_list(), batch_first=True).to(device)
        mask = torch.zeros_like(self.__seq) # Attention mask, so we don't infer on padding
        for i, seq in enumerate(df.tensor):
            mask[i, 0:len(seq)] = 1
        self.__mask = mask 
        
        self.batch = [(self.__seq[b:b+batch_size],
                       self.__mask[b:b+batch_size],
                       self.__labels[b:b+batch_size]) for b in batches]
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.__batch_id)
        for b in self.__batch_id:
            yield self.batch[b]
    
    def __len__(self):
        return self.__len

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to Aicrowd
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