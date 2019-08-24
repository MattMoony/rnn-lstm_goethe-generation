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

    for j, c in enumerate(ascii_lowercase + ascii_lowercase.upper() + ''.join([str(n) for n in range(10)])):
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

    dst = np.array([])
    for ex in dst_raw:
        ex_arr = np.zeros((len(ex), len(dct)))
        for j, c in enumerate(ex):
            for k, v in dct.items():
                if v == c:
                    ex_arr[j, k] = 1
        
        try:
            np.append(dst, ex_arr, axis=1)
        except ValueError:
            dst = ex_arr

    return dst, dct

def save_dataset(dst: np.ndarray, dst_fname: str = 'data.pkl') -> None:
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

def init_lstm_weights(*args):
    for w in args:
        w *= np.random.rand(*w.shape) * (1 / w.shape[-1])

def generate(xs, r_w_s, fc_w_s, b_s, n):
    h = xs[0].shape[0]

    ct = np.zeros((len(r_w_s), h))
    ht = np.zeros((len(r_w_s)+1, h))

    zl = None
    al = None

    for x in xs:
        ht = np.vstack((x, ht[1:]))
        w_i = 0

        for w, actf, actf_g in r_w_s:
            stack = np.vstack((ht[w_i+1], ht[w_i]))
            zt = np.sum(w * stack, 1) + b_s[w_i]
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
            stack = np.vstack((ht[w_i+1], ht[w_i]))
            zt = np.sum(w * stack, 1) + b_s[w_i]
            i, f, o, g = [actf[u](v) for u, v in enumerate(zt)]

            ct[w_i] = ct[w_i] * f + i * g
            ht[w_i+1] = o * actf[-1](ct[w_i])

            w_i += 1

    return ys

def predict(xs, r_w_s, fc_w_s, b_s):
    h = xs[0].shape[0]

    ct = np.zeros((len(r_w_s), h))
    ht = np.zeros((len(r_w_s)+1, h))

    zl = None
    al = None

    ys = None

    for x in xs:
        ht = np.vstack((x, ht[1:]))
        w_i = 0

        for w, actf, actf_g in r_w_s:
            stack = np.vstack((ht[w_i+1], ht[w_i]))
            zt = np.sum(w * stack, 1) + b_s[w_i]
            i, f, o, g = [actf[u](v) for u, v in enumerate(zt)]

            ct[w_i] = ct[w_i] * f + i * g
            ht[w_i+1] = o * actf[-1](ct[w_i])

            w_i += 1

        al = ht[-1]

        for w, actf, actf_g in fc_w_s:
            zl = w.dot(al) + b_s[w_i]
            al = actf(zl)

            w_i += 1

        try:
            ys = np.append(ys, al[np.newaxis,:], axis=0)
        except ValueError:
            ys = al[np.newaxis,:]

    return ys

def loss(xs, ys, r_w_s, fc_w_s, b_s, lamb=1e-6):
    n = len(ys)
    preds = predict(xs, r_w_s, fc_w_s, b_s)
    
    l = np.sum(-np.log(preds[np.arange(n), ys]))
    l /= n

    l += (lamb / (2 * n)) * np.sum(np.asarray([np.sum(u[0] ** 2) for u in r_w_s + fc_w_s]))
    return l

def compute_gradient(xs, ys, r_w_s, fc_w_s, b_s, lamb=1e-6):
    h = xs[0].shape[0]
    n = len(ys)

    ct = np.zeros((len(r_w_s), h))
    ht = np.zeros((len(r_w_s) + 1, h))

    ct_s = np.zeros((n, len(r_w_s), h))
    zt_s = np.zeros((n, len(r_w_s), 4, h))
    ht_s = np.zeros((n, len(r_w_s) + 1, h))

    ct_g = np.zeros((n + 1, len(r_w_s), h))
    st_g = np.zeros((n + 1, len(r_w_s) + 1, 2, h))

    it_s = np.zeros((n, len(r_w_s), h))
    ft_s = np.zeros((n, len(r_w_s), h))
    ot_s = np.zeros((n, len(r_w_s), h))
    gt_s = np.zeros((n, len(r_w_s), h))

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

    cu_ind = 0

    for x in xs:
        ht = np.vstack((x, ht[1:]))
        w_i = 0

        for w, actf, actf_g in r_w_s:
            stack = np.vstack((ht[w_i+1], ht[w_i]))
            zt = np.sum(w * stack, 1) + b_s[w_i]
            i, f, o, g = [actf[u](v) for u, v in enumerate(zt)]

            zt_s[cu_ind][w_i] = zt

            it_s[cu_ind][w_i] = i
            ft_s[cu_ind][w_i] = f
            ot_s[cu_ind][w_i] = o
            gt_s[cu_ind][w_i] = g

            ct[w_i] = ct[w_i] * f + i * g
            ht[w_i+1] = o * actf[-1](ct[w_i])

            w_i += 1

        ct_s[cu_ind]    = ct.copy()
        ht_s[cu_ind]    = ht.copy()

        zl_s.append([])
        al_s.append([])

        al = ht[-1]
        al_s[cu_ind].append(al)

        for w, actf, actf_g in fc_w_s:
            zl = w.dot(al) + b_s[w_i]
            al = actf(zl)

            zl_s[cu_ind].append(zl)
            al_s[cu_ind].append(al)

            w_i += 1

        cu_ind += 1

    # -- BACKWARD PASS ----------------------------------------------- #    

    for cu_ind in reversed(range(n)):
        w_i = len(b_s) - 1
        al_g = np.ones(al.shape)

        for w, actf, actf_g in reversed(fc_w_s):
            zl_g = al_g * actf_g(zl_s[cu_ind][len(b_s) - w_i - 1], ys[cu_ind])

            b_s_g[w_i] += np.sum(zl_g)
            fc_w_s_g[w_i - len(r_w_s)] += zl_g.dot(al_s[cu_ind][len(b_s) - w_i - 1].T)

            al_g = w.T.dot(zl_g)
            w_i -= 1

        st_g[cu_ind][w_i+1] = np.vstack((np.zeros(h), al_g))

        for w, actf, actf_g in reversed(r_w_s):
            h_g = st_g[cu_ind+1][w_i][0] + st_g[cu_ind][w_i+1][1]
            c_g = ct_g[cu_ind+1][w_i] + h_g * ot_s[cu_ind][w_i] * actf_g[-1](ct_s[cu_ind][w_i])

            i_g = c_g * gt_s[cu_ind][w_i]
            f_g = c_g * ct_s[cu_ind-1][w_i]
            o_g = h_g * actf[-1](ct_s[cu_ind][w_i])
            g_g = c_g * it_s[cu_ind][w_i]

            z_g = np.vstack((
                i_g * actf_g[0](zt_s[cu_ind][w_i][0]),
                f_g * actf_g[1](zt_s[cu_ind][w_i][1]),
                o_g * actf_g[2](zt_s[cu_ind][w_i][2]),
                g_g * actf_g[3](zt_s[cu_ind][w_i][3])
            ))

            b_s_g[w_i] += np.sum(z_g, 1)[:,np.newaxis]
            r_w_s_g[w_i] += z_g.reshape(4, 1, h) * np.vstack((ht_s[cu_ind-1][w_i+1], ht_s[cu_ind][w_i])).reshape(1, 2, h)

            ct_g[cu_ind][w_i] = c_g * ft_s[cu_ind][w_i]
            st_g[cu_ind][w_i] = np.sum(w * z_g.reshape(4, 1, h), 0)

            w_i -= 1


    # -- AVERAGING + REGULARIZATION ---------------------------------- #    

    for j in range(len(r_w_s_g)):
        r_w_s_g[j] /= n
        r_w_s_g[j] += (lamb/n) * r_w_s[j][0]

    for j in range(len(fc_w_s_g)):
        fc_w_s_g[j] /= n
        fc_w_s_g[j] += (lamb/n) * fc_w_s[j][0]

    for j in range(len(b_s_g)):
        b_s_g[j] /= n

    # ---------------------------------------------------------------- #    

    return r_w_s_g, fc_w_s_g, b_s_g

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

def sgd(dst, r_w_s, fc_w_s, b_s, lamb=1e-6, alpha=1, iters=32, batch_size=32, decay=0.9, dec_threshold=1e-32, beta=0.9):
    # -- INITIALIZATION ---------------------------------------------- #    

    past_Js = []

    r_w_s = [[w[0].copy(), w[1], w[2]] for w in r_w_s]
    fc_w_s = [[w[0].copy(), w[1], w[2]] for w in fc_w_s]
    b_s = [b.copy() for b in b_s]

    v_r_w_s = [np.zeros(w[0].shape) for w in r_w_s]
    v_fc_w_s = [np.zeros(w[0].shape) for w in fc_w_s]
    v_b_s = [np.zeros(b.shape) for b in b_s]

    # -- STOCHASTIC GRADIENT DESCENT --------------------------------- #    

    for i in range(iters):
        starti = np.random.randint(0,len(dst)-batch_size+1)
        batch = dst[starti:starti+batch_size]

        xs = batch[:-1]
        ys = np.argmax(batch[1:], 1)

        r_w_s_g, fc_w_s_g, b_s_g = compute_gradient(xs, ys, r_w_s, fc_w_s, b_s, lamb)

        for j, w_g in enumerate(r_w_s_g):
            v_r_w_s[j] = beta * v_r_w_s[j] + (1 - beta) * w_g
            r_w_s[j][0] -= alpha * v_r_w_s[j]

        for j, w_g in enumerate(fc_w_s_g):
            v_fc_w_s[j] = beta * v_fc_w_s[j] + (1 - beta) * w_g
            fc_w_s[j][0] -= alpha * v_fc_w_s[j]

        for j, b_g in enumerate(b_s_g):
            v_b_s[j] = beta * v_b_s[j] + (1 - beta) * b_g
            b_s[j] -= alpha * v_b_s[j]

        past_Js.append(loss(xs, ys, r_w_s, fc_w_s, b_s, lamb))

        print('[iter#{:04d}]: loss -> {:} ... '.format(i, past_Js[-1]))

        try:
            if past_Js[-2] - past_Js[-1] <= dec_threshold:
                alpha *= decay
                dec_threshold *= decay
        except IndexError:
            pass

        print('\t...: alpha -> {:} ... '.format(alpha))

    # ---------------------------------------------------------------- #    

    return r_w_s, fc_w_s, b_s, past_Js

def find_hyps(dst, r_w_s, fc_w_s, b_s, alpha_range, lamb_range, n, iters=16, batch_size=32, decay=0.9, dec_threshold=1e-32, beta=0.9):
    alphas  = np.random.rand(n) * (alpha_range[1] - alpha_range[0]) + alpha_range[0]
    lambs   = np.random.rand(n) * (lamb_range[1] - lamb_range[0]) + lamb_range[0]

    l_lss = np.inf
    
    b_alpha = 0
    b_lamb  = 0

    for alpha in alphas:
        for lamb in lambs:
            n_r_w_s, n_fc_w_s, n_b_s, past_Js = sgd(dst, r_w_s, fc_w_s, b_s, 
                lamb=lamb, alpha=alpha, iters=iters, batch_size=batch_size, decay=decay, dec_threshold=dec_threshold, beta=beta)

            starti = np.random.randint(0,len(dst)-batch_size+1)
            batch = dst[starti:starti+batch_size]

            xs = batch[:-1]
            ys = np.argmax(batch[1:], 1)

            c_lss = loss(xs, ys, n_r_w_s, n_fc_w_s, n_b_s) 
            print('[a={:.3f};l={:.3f}] ... loss -> {:} ... '.format(alpha, lamb, c_lss))
            
            if c_lss < l_lss:
                l_lss = c_lss
                
                b_alpha = alpha
                b_lamb = lamb

    return b_alpha, b_lamb

def find_lamb(dst, r_w_s, fc_w_s, b_s, alpha, lamb_range, n, iters=16, batch_size=32, decay=0.9, dec_threshold=1e-32, beta=0.9):
    lambs   = np.random.rand(n) * (lamb_range[1] - lamb_range[0]) + lamb_range[0]

    l_lss = np.inf
    b_lamb  = 0

    for lamb in lambs:
        n_r_w_s, n_fc_w_s, n_b_s, past_Js = sgd(dst, r_w_s, fc_w_s, b_s, 
            lamb=lamb, alpha=alpha, iters=iters, batch_size=batch_size, decay=decay, dec_threshold=dec_threshold, beta=beta)

        starti = np.random.randint(0,len(dst)-batch_size+1)
        batch = dst[starti:starti+batch_size]

        xs = batch[:-1]
        ys = np.argmax(batch[1:], 1)

        c_lss = loss(xs, ys, n_r_w_s, n_fc_w_s, n_b_s) 
        print('[l={:.3f}] ... loss -> {:} ... '.format(lamb, c_lss))
        
        if c_lss < l_lss:
            l_lss = c_lss
            b_lamb = lamb

    return b_lamb

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

def main():
    print('# -- LOAD DATASET ------------------------------------------------ #')

    # dst, dct = build_dataset()
    # save_dataset(dst)
    # return

    dst, dct = read_dataset()
    # dst = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # dct = {
    #     0: 'h',
    #     1: 'e',
    #     2: 'l',
    #     3: 'o'
    # }

    print('# -- SETUP MODEL ------------------------------------------------- #')

    h = len(dct)

    w1 = np.ones((4, 2, h))
    w2 = np.ones((4, 2, h))
    w3 = np.ones((4, 2, h))

    w4 = np.ones((h, h))

    # init_weights(w1, w2, w3, w4)

    init_lstm_weights(w1, w2, w3)
    init_weights(w4)

    b1 = np.zeros((4, 1)) + 1e-8
    b2 = np.zeros((4, 1)) + 1e-8
    b3 = np.zeros((4, 1)) + 1e-8

    b4 = np.zeros(1) + 1e-8

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
        [
            w3,
            (sigmoid, sigmoid, sigmoid, tanh, tanh), 
            (sigmoid_grad, sigmoid_grad, sigmoid_grad, tanh_grad, tanh_grad)
        ]
    ]

    fc_w_s = [
        [
            w4,
            softmax,
            softmax_grad
        ]
    ]

    b_s = [b1, b2, b3, b4]

    print('# -- TRAIN MODEL ------------------------------------------------- #')

    lamb = 3e-5
    alpha = 10
    iters = 512
    batch_size = 128
    decay = 0.9
    dec_threshold = 1e-64
    beta = 0.9

    # alpha, lamb = find_hyps(dst, r_w_s, fc_w_s, b_s, alpha_range=[1e-2, 100], lamb_range=[0,1], n=16,
    #     iters=16, batch_size=batch_size, decay=decay, dec_threshold=dec_threshold, beta=beta)

    lamb = find_lamb(dst, r_w_s, fc_w_s, b_s, alpha, 
        lamb_range=[0,1], n=32, iters=64, batch_size=batch_size, decay=decay, dec_threshold=dec_threshold, beta=beta)

    print('='*32)
    print('Best Lambda: {:} ... '.format(lamb))
    print('Press <ENTER> to continue ... ')
    input('='*32)

    n_r_w_s, n_fc_w_s, n_b_s, past_Js = sgd(dst, r_w_s, fc_w_s, b_s, 
        lamb=lamb, alpha=alpha, iters=iters, batch_size=batch_size, decay=decay, dec_threshold=dec_threshold, beta=beta)

    # n_r_w_s, n_fc_w_s, n_b_s = r_w_s, fc_w_s, b_s

    print('# -- EVALUATE MODEL ---------------------------------------------- #')

    # xs = translate_to_dict_vector('hell', dct)

    # pred = predict(xs, n_r_w_s, n_fc_w_s, n_b_s)
    # print(pred)
    # print(translate_from_dict_vector(pred, dct))

    # ys = np.array([1, 2, 2, 3])
    # lss = loss(xs, ys, r_w_s, fc_w_s, b_s, lamb)

    # print('Loss: {:} ... '.format(lss))

    print('# -- VISUALIZATION ----------------------------------------------- #')

    plot_j_epoch(past_Js)
    plt.show()

    print('# -- GENERATION -------------------------------------------------- #')

    # n = 4
    # xs = translate_to_dict_vector('h', dct)
    # gen = generate(xs, r_w_s, fc_w_s, b_s, n)

    # print('Generated text: "{:}" ... '.format(translate_from_dict_vector(gen, dct)))

    n_gen_chars = 1024

    try:
        beg_ch = input('Enter beginning character: ')
        text = translate_from_dict_vector(generate(translate_to_dict_vector(beg_ch, dct), 
              n_r_w_s, n_fc_w_s, n_b_s, n_gen_chars), dct)

        with open('out.txt', 'w') as f:
            f.write(text)
    except KeyboardInterrupt:
        pass
    
    print('# ---------------------------------------------------------------- #')

if __name__ == '__main__':
    main()