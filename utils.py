import torch

def char_to_tensor(c, dsize=256, device='cuda'):
    t = torch.zeros(dsize).to(device)
    t[ord(c)] = 1
    return t.long()

def chars_to_tensors(cs, dsize=256, device='cuda'):
    t = torch.zeros(len(cs), dsize).to(device)
    t[tuple(torch.arange(len(cs))), tuple([ord(c) for c in cs])] = 1
    return t.long()

def tensor_to_char(t):
    i = torch.argmax(t).item()
    return chr(i)

def tensors_to_chars(ts):
    i_s = tuple(torch.argmax(ts, axis=1))
    return ''.join([chr(i) for i in i_s])