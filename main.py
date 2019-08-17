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
        dct[j] = c

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
        for j, c in enumerate(ex):
            for k, v in dct.items():
                if v == c:
                    ex_arr[j, k] = 1
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

def plot_j_epoch(past_J):
    plt.figure()
    plt.title('J/Epoch-Graph')

    # plt.plot(np.arange(len(past_J)), past_J[:,0], c='darkslategray', linestyle='-', marker='o', markersize=3)
    # plt.plot(np.arange(len(past_J)), past_J[:,1], c='greenyellow', linestyle='--', marker='o', markersize=3)

    plt.plot(np.arange(len(past_J)), past_J, c='darkslategray', linestyle='-', marker='o', markersize=3)

    plt.xlabel('Epoch')
    plt.ylabel('Cost (J)')
    plt.legend(['J(train)', 'J(val)'])

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

def ReLU(z):
    return np.maximum(0, z)

def ReLU_grad(z, *args, **kwargs):
    grad = np.maximum(0, z)
    grad[grad != 0] = 1
    return grad

def no_actf(z):
    return z

def no_actf_grad(z, *args, **kwargs):
    return 1

def grad_check(z, function, lamb=1e-6):
    return (function(z + lamb) - function(z)) / lamb

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

def init_weights(*args):
    for w in args:
        n_avg = (np.prod(w.shape[1:]) + w.shape[0]) / 2
        w *= np.random.rand(*w.shape) * (1 / n_avg)

def predict(xs, r_w_s, fc_w_s, b_s, n):
    h = xs[0].shape[0]

    ct = np.zeros((len(r_w_s), h))
    ht = np.zeros((len(r_w_s)+1, h))

    zl = None
    al = None

    for x in xs:
        ht = np.vstack((x, ht[1:]))
        w_i = 0

        for w, actf, actf_g in r_w_s:
            stack = np.hstack((ht[w_i+1], ht[w_i]))
            zt = w.dot(stack).reshape(4,-1) + b_s[w_i]
            i, f, o, g = [actf[u](v) for u, v in enumerate(zt)]

            ct[w_i] = ct[w_i] * f + i * g
            ht[w_i+1] = o * actf[-1](ct[w_i])

            w_i += 1

    ys = None

    for i in range(n):
        al = ht[-1]
        w_i = len(r_w_s)

        for w, actf, actf_g in fc_w_s:
            zl = w.dot(al) + b_s[w_i]
            al = actf(zl)

            w_i += 1

        try:
            ys = np.append(ys, al[np.newaxis,:], axis=0)
        except ValueError:
            ys = al[np.newaxis,:]
        x = np.where(np.arange(al.shape[0]) == np.argmax(al), 1, 0)

        ht = np.vstack((x, ht[1:]))
        w_i = 0

        for w, actf, actf_g in r_w_s:
            stack = np.hstack((ht[w_i+1], ht[w_i]))
            zt = w.dot(stack).reshape(4,-1) + b_s[w_i]
            i, f, o, g = [actf[u](v) for u, v in enumerate(zt)]

            ct[w_i] = ct[w_i] * f + i * g
            ht[w_i+1] = o * tanh(ct[w_i])

            w_i += 1

    return ys

def loss(xs, ys, r_w_s, fc_w_s, b_s, lamb=1e-6):
    n = len(ys)
    preds = predict(xs, r_w_s, fc_w_s, b_s, n)
    
    l = np.sum(-np.log(preds[np.arange(n), ys]))
    l /= n

    l += (lamb / (2 * n)) * np.sum(np.asarray([np.sum(u[0] ** 2) for u in r_w_s + fc_w_s]))
    return l

def compute_gradient(xs, ys, r_w_s, fc_w_s, b_s, lamb=1e-6, n=1):
    h = xs[0].shape[0]

    ct = np.zeros((len(r_w_s), h))
    ht = np.zeros((len(r_w_s)+1, h))

    ct_s = []
    ht_s = []
    zt_s = []

    ct_g = []
    ht_g = []

    it = ct.copy()
    ft = it.copy()
    ot = ft.copy()
    gt = ot.copy()

    it_s = []
    ft_s = []
    ot_s = []
    gt_s = []

    zl = None
    al = None

    zl_s = []
    al_s = []

    zl_g = None
    al_g = None

    r_w_s_g = [np.zeros(u[0].shape) for u in r_w_s]
    fc_w_s_g = [np.zeros(u[0].shape) for u in fc_w_s]
    b_s_g = [np.zeros(u.shape) for u in b_s]

    # -- FORWARD PASS ------------------------------------------------ #

    for x in xs:
        ht = np.vstack((x, ht[1:]))
        w_i = 0

        zts = []

        for w, actf, actf_g in r_w_s:
            stack = np.hstack((ht[w_i+1], ht[w_i]))
            zt = w.dot(stack).reshape(4,-1) + b_s[w_i]
            j, f, o, g = [actf[u](v) for u, v in enumerate(zt)]

            ct[w_i] = ct[w_i] * f + j * g
            ht[w_i+1] = o * actf[-1](ct[w_i])

            zts.append(zt)

            it[w_i] = j
            ft[w_i] = f
            ot[w_i] = o
            gt[w_i] = g

            w_i += 1

        ht_s.append(ht.copy())
        ct_s.append(ct.copy())

        zt_s.append(zts)

        it_s.append(it.copy())
        ft_s.append(ft.copy())
        ot_s.append(ot.copy())
        gt_s.append(gt.copy())

    for j in range(n):
        al_s.append([])
        zl_s.append([])

        al = ht[-1]
        w_i = len(r_w_s)

        al_s[j].append(al)

        for w, actf, actf_g in fc_w_s:
            zl = w.dot(al) + b_s[w_i]
            al = actf(zl)

            al_s[j].append(al)
            zl_s[j].append(zl)

            w_i += 1

        x = np.where(np.arange(al.shape[0]) == np.argmax(al), 1, 0)

        ht = np.vstack((x, ht[1:]))
        w_i = 0

        zt_s.append([])

        for w, actf, actf_g in r_w_s:
            stack = np.hstack((ht[w_i+1], ht[w_i]))
            zt = w.dot(stack).reshape(4,-1) + b_s[w_i]
            i, f, o, g = [actf[u](v) for u, v in enumerate(zt)]

            ct[w_i] = ct[w_i] * f + i * g
            ht[w_i+1] = o * tanh(ct[w_i])

            zt_s[len(xs)+j].append(zt.reshape(4,-1))

            it[w_i] = i
            ft[w_i] = f
            ot[w_i] = o
            gt[w_i] = g

            w_i += 1

        ht_s.append(ht)
        ct_s.append(ct)

        it_s.append(it.copy())
        ft_s.append(ft.copy())
        ot_s.append(ot.copy())
        gt_s.append(gt.copy())

    # -- BACKWARD PASS ----------------------------------------------- #    

    ht_g = [np.zeros(u.shape) for u in ht_s]
    ct_g = [np.zeros(u.shape) for u in ct_s]

    ht_g.append(ht_g[-1].copy())
    ct_g.append(ct_g[-1].copy())

    cu_ind = len(xs) + n - 2

    for j in range(n):
        al_g = np.ones(ys[cu_ind].shape)
        w_i = len(b_s) - 1

        for w, actf, actf_g in reversed(fc_w_s):
            grad = al_g * actf_g(zl_s[cu_ind][w_i-len(b_s)+1], ys[cu_ind])

            fc_w_s_g[w_i-len(r_w_s)] += grad.dot(al_s[cu_ind][w_i-len(b_s)].T)
            b_s_g[w_i] += np.sum(grad)

            al_g = w.T.dot(grad.T)
            w_i -= 1

        ht_g[cu_ind][w_i+1] = al_g

        for w, actf, actf_g in reversed(r_w_s):
            h_g = ht_g[cu_ind+1][w_i] + ht_g[cu_ind][w_i+1]
            c_g = ct_g[cu_ind+1][w_i] + h_g * ot_s[cu_ind][w_i] * actf_g[-1](ct_s[cu_ind][w_i])

            i_g = c_g * gt_s[cu_ind][w_i]
            f_g = c_g * ct_s[cu_ind-1][w_i]
            o_g = h_g * actf[-1](ct_s[cu_ind][w_i])
            g_g = c_g * it_s[cu_ind][w_i]

            z_g = np.vstack((
                i_g * actf[0](zt_s[cu_ind][w_i][0]),
                f_g * actf[1](zt_s[cu_ind][w_i][1]),
                o_g * actf[2](zt_s[cu_ind][w_i][2]),
                g_g * actf[3](zt_s[cu_ind][w_i][3])
            ))

            r_w_s_g[w_i] += z_g.reshape(-1,1).dot(np.hstack((ht_s[cu_ind-1][w_i], ht_s[cu_ind][w_i-1]))[np.newaxis,:])
            b_s_g[w_i] += np.sum(z_g, 1)[:,np.newaxis]

            ht_g[cu_ind][w_i] = 

            w_i -= 1

        cu_ind -= 1

    # ---------------------------------------------------------------- #    

    return r_w_s_g, fc_w_s_g, b_s_g

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

def sgd(dst, r_w_s, fc_w_s, b_s, lamb=1e-6, alpha=1, iters=32, batch_size=16, decay=0.9, dec_threshold=1e-2, beta=0.9):
    pass

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

def main():
    # -- LOAD DATASET ------------------------------------------------ #

    # dst, dct = build_dataset()
    # save_dataset(dst)
    # return

    # dst, dct = read_dataset()
    dst = np.array([[1, 0], [0, 1]])
    dct = {
        0: 'h',
        1: 'e'
    }

    # -- SETUP MODEL ------------------------------------------------- #

    h = len(dct)

    w1 = np.ones((4 * h, 2 * h))
    w2 = np.ones((4 * h, 2 * h))

    w3 = np.ones((h, h))

    init_weights(w1, w2, w3)

    b1 = np.zeros((4, 1)) + 1e-8
    b2 = np.zeros((4, 1)) + 1e-8

    b3 = np.zeros(1) + 1e-8

    r_w_s = [
        [
            w1, 
            (sigmoid, sigmoid, sigmoid, tanh, tanh), 
            (sigmoid_grad, sigmoid_grad, sigmoid_grad, tanh_grad, tanh_grad)
        ],
        [
            w2,
            (sigmoid, sigmoid, sigmoid, tanh, tanh), 
            (sigmoid_grad, sigmoid_grad, sigmoid_grad, tanh_grad, tanh_grad)
        ],
    ]

    fc_w_s = [
        [
            w3,
            softmax,
            softmax_grad
        ]
    ]

    b_s = [b1, b2, b3]

    # -- TRAIN MODEL ------------------------------------------------- #

    lamb = 0
    alpha = 10
    iters = 128
    batch_size = 4
    decay = 0.9
    dec_threshold = 1e-16
    beta = 0.9

    # -- EVALUATE MODEL ---------------------------------------------- #

    xs = translate_to_dict_vector('h', dct)
    ys = translate_to_dict_vector('e', dct)
    n = 1

    pred = predict(xs, r_w_s, fc_w_s, b_s, n)
    print(pred)
    print(translate_from_dict_vector(pred, dct))

    print(loss(xs, ys, r_w_s, fc_w_s, b_s))

    r_w_s_g, fc_w_s_g, b_s_g = compute_gradient(xs, ys, r_w_s, fc_w_s, b_s, lamb)

    # print(r_w_s_g)
    # print(fc_w_s_g)
    # print(b_s_g)

    # -- VISUALIZATION ----------------------------------------------- #

    # -- GENERATION -------------------------------------------------- #

    # n_gen_chars = 1024

    # try:
    #     beg_ch = input('Enter beginning character: ')
    #     text = translate_from_dict_vector(predict(translate_to_dict_vector(beg_ch, dct), 
    #           n_rw_s, n_fc_w_s, n_b_s, n_gen_chars), dct)

    #     with open('out.txt', 'w') as f:
    #         f.write(text)
    # except KeyboardInterrupt:
    #     pass
    
    # ---------------------------------------------------------------- #

if __name__ == '__main__':
    main()