from scipy.integrate import quad
import numpy as np
from math import pi
from scipy.special import erfcx
from scipy import interpolate
import time
import pickle



def dawson(u):
    assert isinstance(u, np.ndarray)
    # assert u.ndim == 1
    return (np.sqrt(pi) / 2) * erfcx(-u)


def dbl_dawson(x):
    assert isinstance(x, np.ndarray)
    # assert x.ndim == 1
    length = x.size
    shape = x.shape
    x_flatten = x.flatten()
    res = np.empty([length], dtype=np.float)
    for i in range(length):
        res[i] = pi / 4 * quad(lambda u: np.exp(x_flatten[i]**2 - u**2) * erfcx(-u) ** 2, - np.inf, x_flatten[i])[0]
    return res.reshape(shape)


def d_(x):
    return (np.sqrt(pi) / 2) * erfcx(-x)


def dbl_d(x):
    return pi / 4 * quad(lambda u: np.exp(x**2 - u**2) * erfcx(-u) ** 2, - np.inf, x)[0]


class Fun:
    def __init__(self, t_ref=5, l_upper=0.05, v_r=0, v_th=20):
        self.t_ref = t_ref
        self.L = l_upper
        self.v_r = v_r
        self.v_th = v_th
        self.mu = None

    def s_1(self, y, z):
        # start = time.time()
        assert isinstance(y, np.ndarray)
        assert isinstance(z, np.ndarray)
        assert y.ndim == z.ndim
        length = y.size
        i_1 = ((self.v_r * self.L - y) / z).flatten()
        i_2 = ((self.v_th * self.L - y) / z).flatten()
        # print("Integral up_down\t", i_1[0], i_2[0])

        res = np.empty([length], dtype=np.float)
        for j in range(length):
            temp_local = (2 / self.L) * quad(d_, i_1[j], i_2[j])[0]
            res[j] = 1 / (self.t_ref + temp_local)
        self.mu = res
        # print("Integral(u) takes time:\t", time.time() - start)
        return res.reshape(y.shape)

    def s_2(self, y, z):
        # start = time.time()
        assert self.mu is not None
        length = y.size
        i_1 = ((self.v_r * self.L - y) / z).flatten()
        i_2 = ((self.v_th * self.L - y) / z).flatten()

        res = np.empty([length], dtype=np.float)
        for j in range(length):
            temp_local = np.sqrt(((8 / self.L ** 2) * quad(dbl_d, i_1[j], i_2[j])[0]))
            res[j] = temp_local * self.mu[j]
        self.mu = None
        # print("Integral(s) takes time:\t", time.time() - start)
        return res.reshape(y.shape)


def interpolation():
    # activation for u, respect to u_bar and s_bar
    start = time.time()
    y = np.linspace(-1., 5, 50)
    z = np.linspace(0.1, 5, 50)
    yy, zz = np.meshgrid(y, z)
    func = Fun()
    u = func.s_1(yy, zz)
    cv = func.s_2(yy, zz)
    new_func_u = interpolate.interp2d(yy, zz, u, kind='cubic')
    new_func_cv = interpolate.interp2d(yy, zz, cv, kind='cubic')
    print('interpolation costs time: ', time.time()-start)
    pickle_file_u = open('s_1.pkl', 'wb')
    pickle.dump(new_func_u, pickle_file_u)
    pickle_file_cv = open('s_2.pkl', 'wb')
    pickle.dump(new_func_cv, pickle_file_cv)
    # return new_func_u


if __name__ == '__main__':
    interpolation()


