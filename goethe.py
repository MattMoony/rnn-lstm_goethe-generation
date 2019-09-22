import utils
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F

EMBEDDING_DIM = 8
HIDDEN_DIM = 8

class GoetheNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(GoetheNN, self).__init__()

        self.lstm = nn.LSTM(vocab_size, hidden_dim, num_layers=1)
        self.drop = nn.Dropout(p=0.2)
        self.h2out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hc=None, cc=None):
        h = x.float()

        if type(hc) == torch.Tensor and type(cc) == torch.Tensor:
            h, (hn, cn) = self.lstm(h.view(1, 1, -1), (hc, cc))
        else:
            h, (hn, cn) = self.lstm(h.view(1, 1, -1))

        h = self.h2out(h.view(1, -1))
        h = self.drop(h)
        h = F.log_softmax(h, dim=1)
        return h, hn, cn

def build_dataset(p):
    with open(p, 'rb') as f:
        ds = utils.chars_to_tensors(f.read().decode('utf-8'), device='cuda')
    return ds

def batch_train(model, batch):
    # lss = 0.

    # pred, hn, cn = model(batch[0])
    # lss += F.nll_loss(pred, torch.argmax(batch[1]).unsqueeze(dim=0))

    # for i, c in enumerate(batch[1:-1]):
    #     pred, hn, cn = model(c, hn, cn)
    #     lss += F.nll_loss(pred, torch.argmax(batch[i+1]).unsqueeze(dim=0))

    # lss /= len(batch)
    # return lss

    lss = 0.
    pred, hn, cn = model(batch[0])
    
    for i, c in enumerate(batch[1:-1]):
        pred, hn, cn = model(c, hn, cn)
    
    lss = F.nll_loss(pred, torch.argmax(batch[-1]).unsqueeze(dim=0))
    return lss

def train(model, ds, batch_size=32, lr=1e-2, iters=64, iiters=64):
    model.train()

    optimizer   = optim.Adam(model.parameters(), lr=lr)
    starti      = np.random.randint(0, len(ds)-batch_size-iiters)

    for i in range(iters):
        avg_lss = 0.

        for j in range(iiters):
            optimizer.zero_grad()
            batch = ds[starti+j:starti+j+batch_size+1]

            lss = batch_train(model, batch)
            lss.backward()
            optimizer.step()

            avg_lss += lss.item()

        avg_lss /= iiters
        print('[iter#{:04d}]: Loss -> {:.3f} ... '.format(i, avg_lss))

def batch_process(model, batch):
    pred, hn, cn = model(batch[0])
    for e in batch[1:]:
        pred, hn, cn = model(e, hn, cn)
    return pred.squeeze()

def generate(model, ds, vocab_size, nchars, batch_size=32, device='cuda'):
    model.eval()
    content = ''

    starti  = np.random.randint(0, len(ds)-batch_size)
    batch   = ds[starti:starti+batch_size]
    content += utils.tensors_to_chars(batch)

    pred    = batch_process(model, batch)
    pred    = utils.max_one_zeros(pred)

    content += utils.tensor_to_char(pred)
    batch   = torch.cat((batch[1:], pred.unsqueeze(dim=0)))

    for i in range(nchars-1):
        pred    = batch_process(model, batch)
        pred    = utils.max_one_zeros(pred)

        content += utils.tensor_to_char(pred)
        batch   = torch.cat((batch[1:], pred.unsqueeze(dim=0)))

    return content

def main():
    vocab_size  = 256
    JohaNN = GoetheNN(EMBEDDING_DIM, HIDDEN_DIM, vocab_size)
    JohaNN = JohaNN.cuda()

    inp = utils.char_to_tensor('ß', device='cuda')
    cha, hn, cn = JohaNN(inp)

    print('Building dataset ... ')
    ds = build_dataset('data.txt')
    print('Dataset-Size: {}'.format(ds.size()))

    train(JohaNN, ds, iters=64, iiters=256, lr=1e-3, batch_size=32)

    print('='*64)
    samp = generate(JohaNN, ds, vocab_size, nchars=64, batch_size=64)
    print('Sample: {}'.format(samp))
    with open('sample.txt', 'wb') as f:
        f.write(samp.encode('utf-8'))
    print('='*64)

if __name__ == '__main__':
    main()