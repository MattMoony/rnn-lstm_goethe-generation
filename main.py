import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from string import ascii_lowercase

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

def build_dataset(folder_name: str = '.\\goethe') -> (list, dict):
    dst_raw = []
    dct = {}

    for i, c in enumerate(ascii_lowercase + ascii_lowercase.upper() + ''.join([str(n) for n in range(10)])):
        dct[i] = c

    for root, dirs, files in os.walk(folder_name):
        for f_name in files:
            with open(os.path.join(folder_name, f_name), 'r') as f:
                cont = ''.join(f.readlines())
                cont_chars = list(cont)

                while len(cont_chars) > 0:
                    if not (cont_chars[0] in dct.values()):
                        dct[list(dct.keys())[-1]+1] = cont_chars[0]

                    cont_chars = list(filter(lambda x: x != cont_chars[0], cont_chars))

                dst_raw.append(cont)

    dst = []
    for ex in dst_raw:
        ex_arr = np.zeros((len(ex), len(dct)))
        for i, c in enumerate(ex):
            for k, v in dct.items():
                if v == c:
                    ex_arr[i, k] = 1
        dst.append(ex_arr)

    return dst, dct

def save_dataset(dst: list, dst_fname: str = 'data.pkl') -> None:
    with open(dst_fname, 'wb') as f:
        pickle.dump(dst, f)

def read_dataset(dst_fname: str = 'data.pkl', dct_fname: str = 'dict.csv') -> (list, dict):    
    f = open(dst_fname, 'rb')
    dst = pickle.load(f)
    f.close()

    df = pd.read_csv(dct_fname)
    dct = {k: v for k,v in df.values}

    return dst, dct

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

def translate_to_dict_vector(c: str, dct: dict) -> np.ndarray:
    d = len(dct)
    vs = list(dct.values())

    inds = [vs.index(u) for u in c]
    return np.asarray([np.where(np.arange(d) == u, 1, 0) for u in inds])

def translate_from_dict_vector(v: np.ndarray, dct: dict) -> str:
    if len(v.shape) == 1:
        return dct[np.argmax(v)]
    else:
        return ''.join([dct[np.argmax(u)] for u in v])

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(z, *args, **kwargs):
    return sigmoid(z) * (1 - sigmoid(z))

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def tanh_grad(z, *args, **kwargs):
    return (4 * np.exp(z) * np.exp(-z)) / ((np.exp(z) + np.exp(-z)) ** 2)

def softmax(Z):
    Z = Z.copy()
    Z -= np.max(Z)
    return np.exp(Z) / np.sum(np.exp(Z))

def softmax_grad(Z, y, *args, **kwargs):
    grad = softmax(Z)
    grad[y] -= 1
    return grad

def grad_check(z, function, lamb=1e-6):
    return (function(z + lamb) - function(z)) / lamb

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

def predict(xs, r_ws, b_s, n) -> np.ndarray:
    d = xs[0].shape[0]

    ct = np.zeros((len(r_ws), d))
    ht = np.zeros((len(r_ws), d))
    xt = np.zeros((len(r_ws)+1, d))

    y = np.zeros((n+1, d))

    for x in xs:
        w_i = 0
        xt[w_i] = x

        for w, actf, actf_g in r_ws:
            parts = w.dot(np.hstack((ht[w_i], xt[w_i]))).reshape(4,-1)
            parts = (parts.T + b_s[w_i]).T
            i, f, o, g = [actf[n](u) for n, u in enumerate(parts)]
            
            ct[w_i] = ct[w_i] * f + i * g
            ht[w_i] = o * actf[-2](ct[w_i])

            xt[w_i+1] = actf[-1](ht[w_i])
            w_i += 1

    y[0] = xt[-1]

    for j in range(n):
        w_i = 0
        xt[w_i] = np.where(np.arange(d) == np.argmax(y[j]), 1, 0)
        
        for w, actf, actf_g in r_ws:
            parts = w.dot(np.hstack((ht[w_i], xt[w_i]))).reshape(4,-1)
            parts = (parts.T + b_s[w_i]).T
            i, f, o, g = [actf[n](u) for n, u in enumerate(parts)]

            ct[w_i] = ct[w_i] * f + i * g
            ht[w_i] = o * actf[-2](ct[w_i])

            xt[w_i+1] = actf[-1](ht[w_i])
            w_i += 1

        y[j+1] = xt[w_i]
    
    return y

def loss(xs, ys, r_ws, b_s, lamb=1e-6):
    d = xs[0].shape[0]
    n = ys.shape[0]

    ct = np.zeros((len(r_ws), d))
    ht = np.zeros((len(r_ws), d))
    xt = np.zeros((len(r_ws)+1, d))

    y = np.zeros((n+1, d))

    for x in xs:
        w_i = 0
        xt[w_i] = x

        for w, actf, actf_g in r_ws:
            parts = w.dot(np.hstack((ht[w_i], xt[w_i]))).reshape(4,-1)
            parts = (parts.T + b_s[w_i]).T
            i, f, o, g = [actf[n](u) for n, u in enumerate(parts)]
            
            ct[w_i] = ct[w_i] * f + i * g
            ht[w_i] = o * actf[-2](ct[w_i])

            xt[w_i+1] = actf[-1](ht[w_i])
            w_i += 1

    y[0] = xt[-1]
    loss = 0

    for j in range(n):
        loss += np.sum(-np.log(y[j, ys[j]]))

        w_i = 0
        xt[w_i] = np.where(np.arange(d) == np.argmax(y[j]), 1, 0)
        
        for w, actf, actf_g in r_ws:
            parts = w.dot(np.hstack((ht[w_i], xt[w_i]))).reshape(4,-1)
            parts = (parts.T + b_s[w_i]).T
            i, f, o, g = [actf[n](u) for n, u in enumerate(parts)]

            ct[w_i] = ct[w_i] * f + i * g
            ht[w_i] = o * actf[-2](ct[w_i])

            xt[w_i+1] = actf[-1](ht[w_i])
            w_i += 1

        y[j+1] = xt[w_i]

    loss /= n
    loss += (lamb / (2*n)) * np.sum(np.asarray([u[0] for u in r_ws]) ** 2)

    return loss

def compute_gradient(xs, ys, r_ws, b_s, lamb=1e-6):
    # -- VARIABLE DEFINITIONS ---------------------------------------- #
    
    d = xs[0].shape[0]
    n = ys.shape[0]
    
    ct = np.zeros((len(r_ws), d))
    ht = np.zeros((len(r_ws), d))
    xt = np.zeros((len(r_ws)+1, d))
    
    ct_s = []
    ht_s = []
    xt_s = []

    ct_g_s = ct.copy()
    ht_g_s = ht.copy()
    xt_g_s = xt.copy()+1

    rw_g = [np.zeros(w[0].shape) for w in r_ws]
    b_g = [np.zeros(b.shape) for b in b_s]

    ft = np.zeros((len(r_ws), d))
    it = np.zeros((len(r_ws), d))
    gt = np.zeros((len(r_ws), d))
    ot = np.zeros((len(r_ws), d))
    
    ft_s = []
    it_s = []
    gt_s = []
    ot_s = []
    
    y = np.zeros((n+1, d))

    # -- FORWARD PASS ------------------------------------------------ #
    
    for x in xs:
        w_i = 0
        xt[w_i] = x
        
        for w, actf, actf_g in r_ws:
            parts = w.dot(np.hstack((ht[w_i], xt[w_i]))).reshape(4,-1)
            parts = (parts.T + b_s[w_i]).T
            i, f, o, g = [actf[n](u) for n, u in enumerate(parts)]

            it[w_i] = i
            ft[w_i] = f
            ot[w_i] = o
            gt[w_i] = g
            
            ct[w_i] = ct[w_i] * f + i * g
            ht[w_i] = o * actf[-2](ct[w_i])

            xt[w_i+1] = actf[-1](ht[w_i])
            w_i += 1
            
            ct_s.append(ct.copy())
            ht_s.append(ht.copy())
            xt_s.append(xt.copy())
            
            it_s.append(it.copy())
            ft_s.append(ft.copy())
            ot_s.append(ot.copy())
            gt_s.append(gt.copy())

    y[0] = xt[w_i]
            
    for j in range(n):
        w_i = 0
        xt[w_i] = np.where(np.arange(d) == np.argmax(y[j]), 1, 0)
        
        for w, actf, actf_g in r_ws:
            parts = w.dot(np.hstack((ht[w_i], xt[w_i]))).reshape(4,-1)
            parts = (parts.T + b_s[w_i]).T
            i, f, o, g = [actf[n](u) for n, u in enumerate(parts)]
            
            it[w_i] = i
            ft[w_i] = f
            ot[w_i] = o
            gt[w_i] = g
            
            ct[w_i] = ct[w_i] * f + i * g
            ht[w_i] = o * actf[-2](ct[w_i])
            
            xt[w_i+1] = actf[-1](ht[w_i])
            w_i += 1
            
        y[j+1] = xt[w_i]
    
        ct_s.append(ct.copy())
        ht_s.append(ht.copy())
        xt_s.append(xt.copy())
        
        it_s.append(it.copy())
        ft_s.append(ft.copy())
        ot_s.append(ot.copy())
        gt_s.append(gt.copy())
                    
    # -- BACKWARD PASS ----------------------------------------------- #

    for j in reversed(range(n)):
        w_i = len(r_ws) - 1

        for w, actf, actf_g in reversed(r_ws):
            cu_ind = -(j-n+1)-1

            end_g = (
                xt_g_s[w_i] * 
                actf_g[-1](ht_s[cu_ind][w_i], np.argmax(xt_s[cu_ind][w_i])) + 
                ht_g_s[w_i]
            )
            highw_g = (
                end_g *
                ot_s[cu_ind][w_i] *
                actf_g[-2](ct_s[cu_ind][w_i]) +
                ct_g_s[w_i]
            ) 

            ft_g = highw_g * ct_s[cu_ind-1][w_i]
            it_g = highw_g * gt_s[cu_ind][w_i]
            gt_g = highw_g * it_s[cu_ind][w_i]
            ot_g = end_g * actf[-2](ct_s[-(j-n+1)][w_i])

            parts_g = np.vstack((
                it_g * actf_g[0](it_s[cu_ind][w_i]), 
                ft_g * actf_g[1](ft_s[cu_ind][w_i]), 
                ot_g * actf_g[2](ot_s[cu_ind][w_i]), 
                gt_g * actf_g[3](gt_s[cu_ind][w_i])
            ))

            b_g[w_i] += np.sum(parts_g, 1)
            rw_g[w_i] += parts_g.reshape(4*d,-1).dot(np.hstack((ht_s[cu_ind-1][w_i], xt_s[cu_ind-1][w_i]))[np.newaxis,:])

            ct_g_s = highw_g * ft_s[cu_ind][w_i]
            
            stacked = w.T.dot(parts_g.reshape(4*d,-1)).reshape(2,-1)

            ht_g_s = stacked[0]
            xt_g_s = stacked[1]

    b_g = [(b / n) for b in b_g]
    rw_g = [(w / n + (lamb / n) * w) for w in rw_g]

    return rw_g, b_g

    # ---------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

def main():
    # -- LOAD DATASET ------------------------------------------------ #

    # dst, dct = build_dataset()
    # save_dataset(dst)

    # return

    dst, dct = read_dataset()

    # -- SETUP MODEL ------------------------------------------------- #

    h = len(dct)

    w1 = np.random.rand(4 * h, 2 * h)
    b1 = np.random.rand(4) + 1e-6

    r_ws = [(
        w1, 
        (sigmoid, sigmoid, sigmoid, tanh, tanh, softmax), 
        (sigmoid_grad, sigmoid_grad, sigmoid_grad, tanh_grad, tanh_grad, softmax_grad)
    )]
    b_s = [b1]

    # -- TRAIN MODEL ------------------------------------------------- #

    lamb = 1e-6
    alpha = 1
    iters = 16
    
    lss = loss(dst[0][:10], np.argmax(dst[0][11:15], 1), r_ws, b_s, lamb)
    rw_g, b_g = compute_gradient(dst[0][:10], np.argmax(dst[0][11:15], 1), r_ws, b_s, lamb)

    # -- EVALUATE MODEL ---------------------------------------------- #

    pred = predict(translate_to_dict_vector('Hello W', dct), r_ws, b_s, 16)

    # -- VISUALIZATION ----------------------------------------------- #


    # ---------------------------------------------------------------- #

if __name__ == '__main__':
    main()