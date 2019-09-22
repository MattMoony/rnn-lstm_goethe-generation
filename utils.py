import torch

def char_to_tensor(c, dsize=256, device='cuda'):
    t = torch.zeros(dsize).long().to(device)
    t[ord(c)] = 1
    return t

def chars_to_tensors(cs, dsize=256, device='cuda'):
    ords = []
    for c in cs:
        o = ord(c)
        if o < 256:
            ords.append(o)

    t = torch.zeros(len(ords), dsize).long().to(device)
    t[tuple(torch.arange(len(ords))), tuple(ords)] = 1
    return t

def tensor_to_char(t):
    i = torch.argmax(t).item()
    return chr(i)

def tensors_to_chars(ts):
    i_s = tuple(torch.argmax(ts, axis=1))
    return ''.join([chr(i) for i in i_s])

def max_one_zeros(z, device='cuda'):
    return torch.where(torch.arange(len(z)).to(device) == torch.argmax(z),\
                        torch.ones(len(z)).to(device), torch.zeros(len(z)).to(device)).long()