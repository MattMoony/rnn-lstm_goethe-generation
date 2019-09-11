import torch
from torch import nn, optim
import torch.nn.functional as F

EMBEDDING_DIM = 8
HIDDEN_DIM = 8

class GoetheNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, out_size):
        self.letter_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2)
        self.h2out = nn.Linear(hidden_dim, out_size)

    def forward(self, x, hc=None, cc=None):
        embs = self.letter_embeddings(x)
        lstm_out, (hn, cn) = self.lstm(embs.view(1, 1, -1), hc, cc)
        out_raw = self.h2out(lstm_out.view(1, -1))
        out = F.log_softmax(out_raw, dim=1)
        return out

def main():
    pass

if __name__ == '__main__':
    main()