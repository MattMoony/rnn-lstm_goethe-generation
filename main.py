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
    for k, v in dct.items():
        if v == c:
            return np.where(np.arange(len(dct)) == k, 1, 0)

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
    Z -= np.max(Z, 1, keepdims=True)
    return np.exp(Z) / np.sum(np.exp(Z), 1)[:,np.newaxis]

def softmax_grad(Z, y, *args, **kwargs):
    grad = softmax(Z)
    grad[np.arange(grad.shape[0]), y] -= 1
    return grad

def grad_check(z, function, lamb=1e-6):
    return (function(z + lamb) - function(z)) / lamb

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

def predict(xs, r_ws, b_s, n) -> np.ndarray:
    d = xs[0].shape[0]

    ct = np.zeros((len(r_ws), d))
    ht = np.zeros((len(r_ws), d))

    y = np.zeros((n, d))

    for x in xs:
        w_i = 0

        for w, actf_s, actf_g_s in r_ws:
            parts = w.dot(np.hstack((ht[w_i], x))).reshape(4,-1)
            i, f, o, g = [actf_s[n](u) for n, u in enumerate(parts)]
            
            ct[w_i] = ct[w_i] * f + i * g
            ht[w_i] = o * actf_s[-1](ct[w_i])

            w_i += 1

    for j in range(n):
        # y[j] = softmax(ht[-1].reshape(1,-1))
        y[j] = np.where(np.arange(d) == np.argmax(ht[-1]), 1, 0)

        w_i = 0
        
        for w, actf_s, actf_g_s in r_ws:
            parts = w.dot(np.hstack((ht[w_i], y[-1]))).reshape(4,-1)
            i, f, o, g = [actf_s[n](u) for n, u in enumerate(parts)]

            ct[w_i] = ct[w_i] * f + i * g
            ht[w_i] = o * actf_s[-1](ct[w_i])

            w_i += 1
    
    return y

# -------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------- #

def main():
    # -- LOAD DATASET ------------------------------------------------ #

    dst, dct = read_dataset()

    # -- SETUP MODEL ------------------------------------------------- #

    h = len(dct)

    w1 = np.random.rand(4 * h, 2 * h)
    b1 = np.random.rand(4) + 1e-6

    r_ws = [(
        w1, 
        (sigmoid, sigmoid, sigmoid, tanh, tanh), 
        (sigmoid_grad, sigmoid_grad, sigmoid_grad, tanh_grad, tanh_grad)
    )]
    b_s = [b1]

    # -- TRAIN MODEL ------------------------------------------------- #
    


    # -- EVALUATE MODEL ---------------------------------------------- #

    pred = predict(np.array([translate_to_dict_vector('a', dct)]), r_ws, b_s, 10)
    print(translate_from_dict_vector(pred, dct))

    # -- VISUALIZATION ----------------------------------------------- #


    # ---------------------------------------------------------------- #

if __name__ == '__main__':
    main()