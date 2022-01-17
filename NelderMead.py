from numpy import zeros, ones, argsort, argwhere, abs, sum, array, max, mean, sort, zeros_like
from numba import jit
import numba

'''
scipy.optimize.fmin,method='nelder-mead'的预编译并行版本(使用numba实现),函数参考如下
在服务器的jupyter上用的建议谨慎,容易试试就逝世
https://scipy.github.io/devdocs/tutorial/optimize.html#id22
from scipy.optimize import fmin
'''


@jit(parallel=True)
def NelderMead(x0, xerr, MaxIter, InitMin, MulRate):
    '''
    https://zhuanlan.zhihu.com/p/427366733
    Nelder-Mead单纯形法求最小值
    :param x0: 初值
    :param xerr: 最小迭代误差变化
    :param MaxIter: 最大迭代次数
    :param InitMin: 最小的初值,只在x0存在0的时候替换
    :param MulRate: 反射点扩大的倍数
    :return: xmin:结果, fmin:最小的函数值
    '''

    N = len(x0)
    x = zeros((N + 1, N))
    y = zeros((N + 1))
    x[0, :] = x0
    temp = x[0, :]
    temp[temp == 0] = InitMin
    x[0, :] = temp
    for ii in range(N):
        x[ii + 1, :] = x[ii, :] * MulRate
    x_last = zeros_like(x)
    mask = ones((N + 1))
    for loopx in range(MaxIter):
        err = max(abs(x[1:, :] - x[0, :]))
        if err < xerr:
            break
        elif (x == x_last).all():
            break
        else:
            x_last = x

        maskwhere = argwhere((mask == 1))
        for ii in range(len(maskwhere)):
            y[maskwhere[ii][0]] = objf(x[maskwhere[ii][0], :])

        order = argsort(y.reshape(-1))
        y = y[order]

        y = sort(y)
        x = x[order, :]
        m = mean(x[:N, :])
        r = 2 * m - x[N, :]
        f_r = objf(r)
        mask[:] = 0
        mask[-1] = 1
        if (y[0] <= f_r) and f_r < y[N]:
            x[N, :] = r
            continue
        elif f_r < y[0]:
            s = m + 2 * (m - x[N, :])
            if objf(s) < f_r:
                x[N, :] = s
            else:
                x[N, :] = r
            continue
        elif f_r < y[N]:
            c1 = m + (r - m) * 0.5
            if objf(c1) < f_r:
                x[N, :] = c1
                continue
        else:
            c2 = m + (x[N, :] - m) * 0.5
            if objf(c2) < y[N]:
                x[N, :] = c2
                continue

        for jj in range(1, N + 1, 1):
            x[jj, :] = x[0, :] + (x[jj, :] - x[0, :]) * 0.5
            mask[jj] = 1
    xmin = x[0, :]
    fmin = objf(xmin)
    return xmin, fmin


# 设置并行数
numba.set_num_threads(12)


# 需要最小化的目标函数,支持限定的numpy函数
@jit(parallel=True)
def objf(x):
    fitness = sum(abs(x ** 2))
    return fitness


# 初值
x0 = array([300, 100])
# 最小误差改变量
xerr = 10 ** -7
MaxIter = 100
InitMin = 0.00025
MulRate = 1.05
xmin, fmin = NelderMead(x0, xerr, MaxIter, InitMin, MulRate)
print(xmin, fmin)
