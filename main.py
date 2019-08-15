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

def grad_check(z, function, lamb=1e-6):
    return (function(z + lamb) - function(z)) / lamb

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

def init_weights(*args):
    for w in args:
        n_avg = (np.prod(w.shape[1:]) + w.shape[0]) / 2
        w *= np.random.rand(*w.shape) * (1 / n_avg)

def fc_pred(x, fc_w_s, b_s):
    zl = None
    al = x

    w_i = 0

    for w, actf, actf_g in fc_w_s:
        zl = w.dot(al) + b_s[w_i]
        al = actf(zl)

        w_i += 1

    return al

def fc_pred_acts(x, fc_w_s, b_s):
    zls = []
    als = [x]

    w_i = 0

    for w, actf, actf_g in fc_w_s:
        zls.append(w.dot(als[-1]) + b_s[w_i])
        als.append(actf(zls[-1]))

        w_i += 1

    return zls, als

def predict(xs, r_ws, fc_w_s, b_s, n) -> np.ndarray:
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

    y[0] = fc_pred(xt[-1], fc_w_s, b_s[len(r_ws):])
    # y[0] = xt[-1]

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

        y[j+1] = fc_pred(xt[w_i], fc_w_s, b_s[len(r_ws):])
        # y[j+1] = xt[w_i]
    
    return y

def loss(xs, ys, r_ws, fc_w_s, b_s, lamb=1e-6):
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

    y[0] = fc_pred(xt[-1], fc_w_s, b_s[len(r_ws):])
    # y[0] = xt[-1]
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

        y[j+1] = fc_pred(xt[w_i], fc_w_s, b_s[len(r_ws):])
        # y[j+1] = xt[w_i]

    # loss /= n
    # loss += (lamb / (2*n)) * np.sum(np.asarray([u[0] for u in r_ws]) ** 2)

    return loss

def fc_compute_gradient(x, y, fc_w_s, b_s, zls, als):
    fc_grads = []
    b_grads = []

    al_grad = np.ones((als[-1].shape))
    w_i = len(fc_w_s)-1

    for w, actf, actf_g in reversed(fc_w_s):
        ac_grad = al_grad * actf_g(zls[w_i], y)

        b_grads.append(np.sum(ac_grad))
        fc_grads.append(als[w_i].T.dot(ac_grad))

    return (fc_w_s[0][0].T.dot(ac_grad)), fc_grads, b_grads

def compute_gradient(xs, ys, r_ws, fc_w_s, b_s, lamb=1e-6):
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
    fc_w_g = [np.zeros(w[0].shape) for w in fc_w_s]
    b_g = [np.zeros(b.shape) for b in b_s]

    zls_t = []
    als_t = []

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

    # y[0] = fc_pred(xt[w_i], fc_w_s, b_s[len(r_ws):])

    zls, als = fc_pred_acts(xt[w_i], fc_w_s, b_s[len(r_ws):])
    y[0] = als[-1]

    zls_t.append(zls)
    als_t.append(als)

    # y[0] = xt[w_i]
            
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
            
        # y[j+1] = fc_pred(xt[w_i], fc_w_s, b_s[len(r_ws):])

        zls, als = fc_pred_acts(xt[w_i], fc_w_s, b_s[len(r_ws):])
        y[j+1] = als[-1]

        zls_t.append(zls)
        als_t.append(als)

        # y[j+1] = xt[w_i]
    
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
        cu_ind = -(j-n+1)-1

        xt_g_s[w_i], fullyc_g, bia_g = fc_compute_gradient(xt_s[cu_ind][w_i], ys[j], fc_w_s, b_s[len(r_ws):], 
            zls_t[cu_ind], als_t[cu_ind])

        for u in range(len(fc_w_g)):
            fc_w_g[u] += fullyc_g[u]
        for u in range(len(b_s[len(r_ws):])):
            b_s[len(r_ws)+u] += bia_g[u]

        for w, actf, actf_g in reversed(r_ws):
            end_g = (
                xt_g_s[w_i] * 
                actf_g[-1](ht_s[cu_ind][w_i], ys[j]) + 
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

            ct_g_s[w_i] = highw_g * ft_s[cu_ind][w_i]
            
            stacked = w.T.dot(parts_g.reshape(4*d,-1)).reshape(2,-1)

            ht_g_s[w_i] = stacked[0]
            xt_g_s[w_i-1] = stacked[1]

            w_i -= 1

    for j, x in enumerate(reversed(xs)):
        w_i = len(r_ws) - 1
        cu_ind = -(n+j)

        for w, actf, actf_g in reversed(r_ws):
            end_g = (
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

            ct_g_s[w_i] = highw_g * ft_s[cu_ind][w_i]
            
            stacked = w.T.dot(parts_g.reshape(4*d,-1)).reshape(2,-1)

            ht_g_s[w_i] = stacked[0]
            xt_g_s[w_i-1] = stacked[1]

            w_i -= 1

    # b_g = [(b / n) for b in b_g]
    # rw_g = [(w / n + (lamb / n) * w) for w in rw_g]

    return rw_g, fc_w_g, b_g

    # ---------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

def sgd(dst, rw_s, fc_w_s, b_s, lamb=1e-6, alpha=1, iters=32, batch_size=16, decay=0.9, dec_threshold=1e-2, beta=0.9, xn=10):
    v_inds = np.arange(len(dst))
    past_Js = []

    d = dst[0].shape[1]

    v_rw_s = []
    v_fc_w_s = []
    v_b_s = []

    rw_s = rw_s.copy()
    for i in range(len(rw_s)):
        rw_s[i][0] = rw_s[i][0].copy()
        v_rw_s.append(np.zeros(rw_s[i][0].shape))
    fc_w_s = fc_w_s.copy()
    for i in range(len(fc_w_s)):
        fc_w_s[i][0] = fc_w_s[i][0].copy()
        v_fc_w_s.append(np.zeros(fc_w_s[i][0].shape))
    b_s = b_s.copy()
    for i in range(len(b_s)):
        b_s[i] = b_s[i].copy()
        v_b_s.append(np.zeros(b_s[i].shape))

    rw_g_s_b = [np.zeros(w[0].shape) for w in rw_s]
    fc_w_g_s_b = [np.zeros(w[0].shape) for w in fc_w_s]
    b_g_s_b = [np.zeros(b.shape) for b in b_s]

    for i in range(iters):
        vi = np.random.choice(v_inds)

        batchi = np.random.choice(np.arange(dst[vi].shape[0]-1-xn), batch_size)
        batch = [(dst[vi][j:j+xn+1], np.argmax(dst[vi][j+xn+1])) for j in batchi]

        tloss = 0.0

        rw_g_s = rw_g_s_b.copy()
        fc_w_g_s = fc_w_g_s_b.copy()
        b_g_s = b_g_s_b.copy()

        for x, y in batch:
            # x = x[np.newaxis, :]
            y = np.array([y])[np.newaxis, :]

            cu_rw_g_s, cu_fc_w_g_s, cu_b_g_s = compute_gradient(x, y, rw_s, fc_w_s, b_s, lamb)

            for j, w_g in enumerate(cu_rw_g_s):
                rw_g_s[j] += w_g
            for j, w_g in enumerate(cu_fc_w_g_s):
                fc_w_g_s[j] += w_g
            for j, b_g in enumerate(cu_b_g_s):
                b_g_s[j] += b_g

            tloss += loss(x, y, rw_s, fc_w_s, b_s, lamb)
        
        for j, w_g in enumerate(rw_g_s):
            w_g /= batch_size
            w_g += (lamb / batch_size) * rw_s[j][0]

            v_rw_s[j] = beta * v_rw_s[j] + (1 - beta) * w_g
            rw_s[j][0] -= alpha * v_rw_s[j]
        for j, w_g in enumerate(fc_w_g_s):
            w_g /= batch_size
            w_g += (lamb / batch_size) * fc_w_s[j][0]

            v_fc_w_s[j] = beta * v_fc_w_s[j] + (1 - beta) * w_g
            fc_w_s[j][0] -= alpha * v_fc_w_s[j]
        for j, b_g in enumerate(b_g_s):
            b_g /= batch_size

            v_b_s[j] = beta * v_b_s[j] + (1 - beta) * b_g
            b_s[j] -= alpha * v_b_s[j]
        
        tloss /= batch_size
        tloss += (lamb / batch_size) * np.sum([np.sum(u[0] ** 2) for u in rw_s])

        past_Js.append(tloss)
        print('[iter#{:04d}]: Loss -> {:}'.format(i, tloss))
        print('[  - | -  ]: Alpha -> {:}'.format(alpha))

        try:
            if abs(past_Js[-1] - past_Js[-2]) >= dec_threshold:
                alpha *= decay
        except IndexError:
            pass

    return rw_s, fc_w_s, b_s, past_Js

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

    w1 = np.ones((4 * h, 2 * h))
    w2 = np.ones((4 * h, 2 * h))

    w3 = np.ones((h, h))

    init_weights(w1, w2, w3)

    b1 = np.zeros(4) + 1e-6
    b2 = np.zeros(4) + 1e-6

    b3 = np.zeros(1) + 1e-6

    rw_s = [
        [
            w1, 
            (sigmoid, sigmoid, sigmoid, tanh, tanh, sigmoid), 
            (sigmoid_grad, sigmoid_grad, sigmoid_grad, tanh_grad, tanh_grad, sigmoid_grad)
        ],
        [
            w2,
            (sigmoid, sigmoid, sigmoid, tanh, tanh, sigmoid), 
            (sigmoid_grad, sigmoid_grad, sigmoid_grad, tanh_grad, tanh_grad, sigmoid_grad)
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

    lamb = 1e-2
    alpha = 10
    iters = 256
    batch_size = 16
    decay = 0.9
    dec_threshold = 0.01
    beta = 0.9
    xn = 8

    n_rw_s, n_fc_w_s, n_b_s, past_Js = sgd(dst, rw_s, fc_w_s, b_s, 
        lamb=lamb, alpha=alpha, iters=iters, batch_size=batch_size, decay=decay, dec_threshold=dec_threshold, beta=beta, xn=xn)

    # -- EVALUATE MODEL ---------------------------------------------- #

    print('[...] Average loss: {:}'.format(np.mean(past_Js)))

    pred = predict(translate_to_dict_vector('Hello W', dct), n_rw_s, n_fc_w_s, n_b_s, 16)
    print(pred)
    print(translate_from_dict_vector(pred, dct))

    pred = predict(translate_to_dict_vector('J', dct), n_rw_s, n_fc_w_s, n_b_s, 16)
    print(translate_from_dict_vector(pred, dct))

    # -- VISUALIZATION ----------------------------------------------- #

    plot_j_epoch(past_Js)
    plt.show()

    # -- GENERATION -------------------------------------------------- #

    n_gen_chars = 1024

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