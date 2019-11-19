import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def load_model(checkpoint_path, model, optimizer):
    try:
        checkpoint = torch.load(checkpoint_path)
        nb_epochs_done = checkpoint['nb_epochs_done']
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f'Successfully loaded {nb_epochs_done} epochs, resuming...')
        return nb_epochs_done
    except FileNotFoundError:
        print('Starting from scratch')
        return 0


def save_model(checkpoint_path, model, optimizer, epoch, embedding_dim):
    checkpoint = {
        'nb_epochs_done': epoch+1,
        'embedding_dim': embedding_dim,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    torch.save(checkpoint, f'{checkpoint_path}_{epoch}.pth')


class BaselineGRU(nn.Module):
    def __init__(self, embedding_dim, vocab_vec, device, bidirectional=True):
        super().__init__()
        self.hidden_dim_gru = 256
        self.embedding_dim, self.bidirectional, self.device = embedding_dim, bidirectional, device
        self.embedding = nn.Embedding.from_pretrained(vocab_vec)
        self.gru = nn.GRU(self.embedding_dim,
                          self.hidden_dim_gru, bidirectional=bidirectional)
        self.classifier = nn.Sequential(nn.Linear(self.hidden_dim_gru*2, 512), nn.ReLU(), nn.Dropout(),
                                        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(),
                                        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(),
                                        nn.Linear(256, 2), nn.LogSoftmax(dim=1))
        #self.h0 = nn.Parameter(torch.zeros(2, 512, self.hidden_dim_gru).type(torch.FloatTensor), requires_grad=True)

    def forward(self, seq, lengths):
        embeddings = pack_padded_sequence(
            self.embedding(seq), lengths).to(self.device)
        h0 = torch.zeros((2, seq.size(1), self.hidden_dim_gru),
                         dtype=torch.float).to(self.device)
        _, h = self.gru(embeddings, h0)
        return self.classifier(h.transpose(0, 1).reshape(-1, self.hidden_dim_gru * 2))
