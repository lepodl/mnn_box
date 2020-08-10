import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from utils import Fun
import time
from opt_einsum import contract


def data_iter(batch_size, features, labels):
    num_examples = features.shape[0]
    index = list(range(num_examples))
    random.shuffle(index)
    for i in range(0, num_examples, batch_size):
        j = index[i: min(i + batch_size, num_examples)]
        yield features[j], labels[j]

def main():
    file = open('./s_1.pkl', 'rb')
    new_s_1 = pickle.load(file)
    file.close()
    with open('./s_2.pkl', 'rb') as f:
        new_s_2 = pickle.load(f)
    x = np.linspace(-1, 4, 100)
    y = np.linspace(5, 15, 100)
    func = Fun()
    u_1 = func.s_1(x, y)
    cv_1 = func.s_2(x, y)
    u_2 = new_s_1(x, y)
    # zipped = zip(x, y)
    cv_2 = np.array(list(map(new_s_2, x, y))).squeeze()
    u_2 = np.diagonal(u_2)
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(np.arange(len(u_1)), u_1 - u_2)
    plt.subplot(122)
    plt.plot(np.arange(len(cv_1)), cv_1 - cv_2)

    plt.show()


def test_map_and_list():
    file = open('./s_1.pkl', 'rb')
    new_s_1 = pickle.load(file)
    file.close()
    # x = np.linspace(-1, 4, 5000)  # [-2, 5, 140]
    # y = np.linspace(5, 15, 5000)  # [2, 20, 180]
    # s1 = time.time()
    # u_1 = np.diagonal(new_s_1(x, y))
    # print("test_direct takes time:", time.time()-s1)
    # s2 = time.time()
    # u_11 = [new_s_1(x[i], y[i]) for i in range(5000)]
    # print("test_direct takes time:", time.time() - s2)
    # s3 = time.time()
    # u_11 = list(map(new_s_1, x, y))
    # print("test_direct takes time:", time.time() - s3)

    x = np.linspace(-3, 2, 10)
    y = np.linspace(5, 10, 10)
    res = new_s_1(x, y)
    print(res[0,1])


def violecity_test():
    W = np.random.uniform(size=(300, 756))
    X = np.random.normal(size=(512, 756, 2))
    rou = np.eye(756)
    rrou = np.expand_dims(rou, 0).repeat(512, axis=0)
    ratio = 0.4
    start = time.time()
    # s = np.sqrt(np.einsum('im,km,in,kn,kmn->ki', W, X[:, :, 1], W, X[:, :, 1], rrou) * (1 + ratio ** 2))
    # s = np.sqrt(np.einsum('im,km,im,km->ki', W, X[:, :, 1], W, X[:, :, 1]) * (1 + ratio ** 2))
    # u = np.einsum('ij,kj->ki', W, X[:, :, 0]) * (1 - ratio)
    dl_gamma = np.einsum('bij,bij->ij', X, X)
    # print(u[1, :5])
    end = time.time()
    print('time for einsum:', end - start)
    start2 = time.time()
    # s = np.sqrt(contract('im,km,in,kn,kmn->ki', W, X[:, :, 1], W, X[:, :, 1], rrou) * (1 + ratio ** 2))
    # s = np.sqrt(contract('im,km,im,km->ki', W, X[:, :, 1], W, X[:, :, 1]) * (1 + ratio ** 2))
    # uu = contract('ij,kj->ki', W, X[:, :, 0]) * (1 - ratio)
    dl_gamma = contract('bij,bij->ij', X, X)
    # print(uu[1, :5])
    end2 = time.time()
    print('time for contract:', end2 - start2)

def test_ddwasomn():
    with open('./dbl_dawson.pkl', 'rb') as f:
        func = pickle.load(f)
    x = np.linspace(-1, 1, 10)
    xx = np.array([[1., 1.5], [-1., 0.2]])
    y = func(x)
    yy = func(xx)
    print(type(y), y)
    print(type(yy), yy)


if __name__ == '__main__':
    x = np.array([1,2,3,4], dtype=np.float)
    y = np.array([1,3,4,3], dtype=np.int)
    print(np.mean(x == y))