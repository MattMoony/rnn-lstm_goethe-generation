import utils
import torch
from torch import nn, optim
import torch.nn.functional as F

EMBEDDING_DIM = 8
HIDDEN_DIM = 8

class GoetheNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(GoetheNN, self).__init__()

        self.letter_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(vocab_size * embedding_dim, hidden_dim, num_layers=2)
        self.h2out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hc=None, cc=None):
        embs = self.letter_embeddings(x)

        if hc and cc:
            lstm_out, (hn, cn) = self.lstm(embs.view(1, 1, -1), (hc, cc))
        else:
            lstm_out, (hn, cn) = self.lstm(embs.view(1, 1, -1))

        out_raw = self.h2out(lstm_out.view(1, -1))
        out = F.log_softmax(out_raw, dim=1)
        return out, hn, cn

def build_dataset(p):
    with open(p, 'rb') as f:
        ds = utils.chars_to_tensors(list(f.read().decode('utf-8')))
    return ds

def main():
    vocab_size  = 256
    JohaNN = GoetheNN(EMBEDDING_DIM, HIDDEN_DIM, vocab_size)
    JohaNN = JohaNN.cuda()

    inp = utils.char_to_tensor('ß')
    cha, hn, cn = JohaNN(inp)

    print('Building dataset ... ')
    ds = build_dataset('data.txt')

    print('='*64)
    print(utils.tensor_to_char(inp))
    print(utils.tensor_to_char(cha), '({})'.format(torch.argmax(cha)))
    print('='*64)

if __name__ == '__main__':
    main()