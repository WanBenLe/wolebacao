'''
Copyright <2021> <Ben Wan: wanbenfighting@gmail.com>
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

PLEASE DO NOT USE FOR ACADEMIC PURPOSES
PLEASE DO NOT USE FOR ACADEMIC PURPOSES
PLEASE DO NOT USE FOR ACADEMIC PURPOSES
'''
import numpy as np
from pickle import load
import pandas as pd
from numpy.linalg import inv
from numba import jit
import time
from numpy.random import choice


@jit(forceobj=True)
def ZScoreCatCon(data, cat, rank):
    xx = 0
    for i in range(data.shape[1]):
        col_temp = data[:, i]
        if (i + 1) in cat:
            kind = np.unique(col_temp)
            temp = np.zeros((len(data), len(kind)))
            for j in range(len(kind)):
                temp[col_temp == kind[j], j] = 1
                temp[:, j] = (temp[:, j] - np.mean(temp[:, j])) / np.std(temp[:, j]) * rank[i] / len(kind)
        else:
            temp = ((col_temp - np.mean(col_temp)) / np.std(col_temp) * rank[i]).reshape(-1, 1)
        if xx == 0:
            result = temp
            xx = 1
        else:
            result = np.hstack((result, temp))

    return result


@jit(fastmath=True, parallel=True)
def meanAxis(matrix):
    result = np.zeros((matrix.shape[1]))
    for i in range(matrix.shape[1]):
        result[i] = np.mean(matrix[:, i])
    return result


@jit(fastmath=True, parallel=True)
# 协方差并行计算
def cov_parallel(cov):
    # Erich Schubert, 2018. Numerically Stable Parallel Computation of (Co-)Variance, SSDBM
    n = len(cov) // 2
    shapex = cov.shape[1]
    all_cov = np.zeros((shapex, shapex))
    V_x = np.zeros((2, shapex))
    v_x = np.zeros((2, shapex))
    # 类似这种的是因为numba加速不支持2-D的运算(axis=~)
    meanx = meanAxis(cov)
    for i in range(shapex):
        cov[:, i] -= meanx[i]
    A = cov[0:n]
    B = cov[n:]
    og_A = len(A) * 0.5
    og_B = len(B) * 0.5
    og_AB = og_A + og_B
    for i in range(shapex):
        V_x[0, i] = np.sum(A[:, i] * 0.5)
        V_x[1, i] = np.sum(B[:, i] * 0.5)
    v_x[0:] = V_x[0:] / og_A
    v_x[1:] = V_x[1:] / og_B

    for i in range(shapex):
        for j in range(shapex):
            VW_A = np.sum(0.5 * ((A[:, i]) - (1 / og_A) * V_x[0, i]) * ((A[:, j]) - (1 / og_A) * V_x[0, j]))
            VW_B = np.sum(0.5 * ((B[:, i]) - (1 / og_B) * V_x[1, i]) * ((B[:, j]) - (1 / og_B) * V_x[1, j]))
            all_cov[i, j] = (VW_A + VW_B + (og_A * og_B) / og_AB * (v_x[0, i] - v_x[1, i]) * (
                    v_x[0, j] - v_x[1, j])) / (og_AB - 1)

    return all_cov


@jit(fastmath=True, parallel=True)
# 组欧式距离
def Edis(mat):
    shape = mat.shape[0]
    dis_mat = np.zeros((shape, shape))
    for i in range(shape):
        for j in range(shape):
            dis_mat[i, j] = (np.mean((mat[i, :] - mat[j, :]) ** 2)) ** 0.5
    return dis_mat


# 求马氏距离
# @jit()
def mdis(data_new, data_merge):
    meanx = meanAxis(data_merge)
    data_new -= meanx
    # cov并行
    A = cov_parallel(data_merge)
    try:
        B = np.linalg.inv(A)
    # except这部分,不属于马氏距离的,一般来说要去掉
    except:
        B = np.linalg.pinv(A)
    # 公式要开根号不能除以100之类的,运算不影响结果不作修改
    D2 = (data_new @ B @ data_new.transpose()) / 100
    return D2


# 根据批中心的马氏距离做比较优势抽样
# @jit(forceobj=True)
def Mdis_Group(data, group, times_all):
    '''
    我也不知道这有没有别人写过,没有就当做是我设计的model吧doge,因此请勿用于学术用途
    1.给定data,抽样的组数,重要性权重和抽样次数将data根据马氏距离尽可能均分成抽样的组数,可用于控制AB测试的测试组和对照组
    2.截断样本至整分的,归一后然后根据重要性加权,分类用onehot
    3.随机取一个batch作为初始分组
    4.然后每次取一个batch计算质心和所有组样本的马氏距离,如果inv裂开就用pinv
    5.取距离最小的组分过去
    '''
    alter = len(data)
    n_sample = int(alter / group / times_all)
    # 对样本取整方便计算
    # data_x预分配了内存,方便抽样,基于同样的原因该部分代码写在并行外面
    data_x = np.zeros((group * n_sample * times_all))
    set_n = [[[]] * group][0]
    # 先随机分配样本給每个组
    sample = choice(np.argwhere(data_x == 0).reshape(-1), group * n_sample, replace=False). \
        reshape(group, n_sample).tolist()
    for i in range(group):
        data_x[sample[i]] = i + 1
        set_n[i].extend(sample[i])
    for i in range(int(times_all - 1)):
        # 随机抽2个样本,分别拼接到AB,计算出4个马氏距离(的平方)A1,B1,A2,B2
        # replace=False是不放回抽样
        Mdis_mat = np.zeros((group, group))

        sample = choice(np.argwhere(data_x == 0).reshape(-1), group * n_sample, replace=False). \
            reshape(group, n_sample).tolist()
        # 第j组样本
        for j in range(group):
            # 第k个数据集
            for k in range(group):
                # 因为set1,set2是动态的所以要用copy
                # 批样本质心
                cluster = np.mean(data[sample[j]], axis=0) - np.mean(data[data_x == k], axis=0)
                temp1 = set_n[k].copy()
                temp1.extend(sample[j])
                Mdis_mat[j, k] = mdis(cluster, data[temp1])
        Mdis_mat = np.round(Mdis_mat, 6)
        resultx = np.zeros((group))
        for j in range(group):
            indx = np.argmin(Mdis_mat[j])
            resultx[j] = indx
            Mdis_mat[:, indx] = 10 ** 99
        for j in range(group):
            set_n[int(resultx[j])].extend(sample[j])
            data_x[sample[j]] = resultx[j] + 1
    result = np.hstack((data, data_x.reshape(-1, 1)))
    return result, set_n


def absdiff(data):
    groupnum = np.unique(data[:, -1])
    result = np.zeros((len(groupnum), len(groupnum)))
    for i in range(len(groupnum)):
        temp1 = data[data[:, -1] == groupnum[i], :-1]
        for j in range(len(groupnum)):
            temp2 = data[data[:, -1] == groupnum[j], :-1]
            result[i, j] = np.sum(np.abs(temp1 - temp2))
    return result


# 随机样本量
alter = 10000
# 特征数
features = 6
# 组数
group = 2
# 抽样次数
times_all = 10
# 随机生成样本
data = np.random.randint(high=13, low=10, size=(alter, features)).astype(float)

# 为分类样本的列
cat = [3]

rank_rate = [1] * features
init_index = np.arange(features).tolist()
n_sample = int(len(data) / group / times_all) * group * times_all
datax = data[:n_sample, :]
deal_data = ZScoreCatCon(datax, cat, rank_rate).astype(float)

# data=data[]


t1 = time.time()
print('start')
result, setx = Mdis_Group(deal_data, group, times_all)

t2 = time.time()
print('运行时间:', t2 - t1)
check_series = np.zeros((group, result.shape[1] - 1))
for i in range(group):
    check_series[i, :] = np.mean(result[result[:, -1] == i + 1][:, :-1], axis=0)

emat = Edis(check_series)
print('重要性加权标化后各组欧氏距离矩阵', emat)
print('重要性加权标化后各组各特征欧氏距离均值', np.mean(emat, axis=1))

# print(std)
# print(inter)
a = absdiff(result)
print(a)

