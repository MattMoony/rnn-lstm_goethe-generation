import os
import utils
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from argparse import ArgumentParser

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

def train(model, ds, batch_size=32, lr=1e-2, iters=64, iiters=64, rand_start=False):
    model.train()

    optimizer   = optim.Adam(model.parameters(), lr=lr)
    if rand_start:
        starti  = np.random.randint(0, len(ds)-batch_size-iiters)
    else:
        starti  = 0

    try:
        for i in range(iters):
            avg_lss = 0.
            print('{:04d}'.format(i), '-'*59)

            for j in range(iiters):
                optimizer.zero_grad()
                batch = ds[starti+j:starti+j+batch_size+1]

                lss = batch_train(model, batch)
                avg_lss += lss
                # lss.backward()
                # optimizer.step()

                # print('\t[{:04d}|{:04d}]: Loss -> {:.3f} ... '.format(i, j, lss.item()))

            avg_lss /= iiters
            avg_lss.backward()
            optimizer.step()

            print('[iter#{:04d}]: Loss -> {:.3f} ... '.format(i, avg_lss.item()))
    except KeyboardInterrupt:
        pass

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
    parser = ArgumentParser()

    parser.add_argument('-p', '--m-path', type=str, dest='mpath', 
                        help='Specify model path ... ')

    parser.add_argument('-g', '--generate', action='store_true', dest='gen',
                        help='Only generate text?')
    parser.add_argument('-n', '--n-chars', type=int, dest='nchars',
                        help='Specify number of chars to be generated ... ', default=64)
    parser.add_argument('--samp-path', type=str, dest='samp_path',
                        help='Specify the sample save path ... ')

    parser.add_argument('--iters', type=int, dest='iters', 
                        help='Specify iteration count ... ', default=128)
    parser.add_argument('--iiters', type=int, dest='iiters',
                        help='Specify inner iteration count ... ', default=1024)
                        
    parser.add_argument('--lr', type=float, dest='lr',
                        help='Specify learning rate (alpha) ... ', default=1e-3)
    parser.add_argument('--batch-size', type=int, dest='batch_size', 
                        help='Specify the batch size ... ', default=32)
    parser.add_argument('--rand-start', action='store_true', dest='rand_start', 
                        help='Random start index?')

    args = parser.parse_args()

    mpath       = args.mpath
    vocab_size  = 256

    JohaNN = GoetheNN(EMBEDDING_DIM, HIDDEN_DIM, vocab_size)
    JohaNN = JohaNN.cuda()

    if args.mpath:
        if os.path.isfile(args.mpath):
            print('Loading model ... ')
            JohaNN.load_state_dict(torch.load(mpath))
        else:
            print('[!] Warning: Model path invalid (no such file); Proceeding without loading a model ... ')

    print('Building dataset ... ')
    ds = build_dataset('data.txt')
    print('Dataset-Size: {}'.format(ds.size()))

    if args.gen:
        print('='*64)
        samp = generate(JohaNN, ds, vocab_size, nchars=args.nchars, batch_size=args.batch_size)
        print('Sample: {}'.format(samp))
        if args.samp_path and os.path.isdir(os.path.dirname(args.samp_path)):
            with open(args.samp_path, 'wb') as f:
                f.write(samp.encode('utf-8'))
        print('='*64)

        os._exit(0)

    train(JohaNN, ds, iters=args.iters, iiters=args.iiters, lr=args.lr, 
        batch_size=args.batch_size, rand_start=args.rand_start)

    print('='*64)
    samp = generate(JohaNN, ds, vocab_size, nchars=128, batch_size=args.batch_size)
    print('Sample: {}'.format(samp))
    if args.samp_path and os.path.isdir(os.path.dirname(args.samp_path)):
            with open(args.samp_path, 'wb') as f:
                f.write(samp.encode('utf-8'))
    print('='*64)

    yN = input('Save model? [y/N] ')
    if yN in ['y', 'Y']:
        spath = ''
        while not os.path.isdir(os.path.dirname(spath)):
            spath = input('Enter destination: ')
        
        torch.save(JohaNN.state_dict(), spath)

if __name__ == '__main__':
    main()