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
from statsmodels.api import OLS, add_constant
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LassoCV
import scipy.stats as stats
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class model_fit():
    def res2ARCHtest(self):
        '''
        Engle, 1982. Autoregressive Conditional Heteroskedasticity With Estimates of the Variance of
         U.K. Inflation, Econometrica
        '''
        x = self.resid2.reshape(-1, 1)
        X = np.vstack((np.zeros((1, 1)), x[0:-1]))
        X3 = add_constant(X)  # 加入常数项
        result3 = OLS(x, X3).fit()  # 回归
        # 经典的nR^2~Chi2(n-k-1)
        Stat = result3.rsquared * len(x - 1 - 1)
        p = stats.chi2.sf(np.abs(Stat) * 2, 1)
        # print('ARCH检验统计量与p值', Stat, p)
        return p

    def dslasso(self):
        '''
        Guanhao Feng, 2020. Taming the Factor Zoo, Journal of Finance
        '''
        result1 = LassoCV(cv=5).fit(self.x, self.y.reshape(-1)).coef_
        # 第一个非0集
        self.Set1 = result1 != 0
        Set2x = np.zeros((len(Set1), self.x.shape[1]))
        index1 = np.argwhere(Set1 == True)[:, 0]
        for i in index1:
            y_ds = self.x[:, i].reshape(-1, 1)
            selct = np.array(range(self.x.shape[1])) != i
            x_ds = self.x[:, selct]
            para_ds = LassoCV(cv=5).fit(x_ds, y_ds.reshape(-1)).coef_
            Set2x[i, selct] = (para_ds != 0).reshape(-1)
        result_st = np.sum(Set2x, axis=0)
        result_sr = np.sum(Set2x, axis=0) / len(index1)
        # print('DS Lasso select time:', result_st)
        # print('DS Lasso ratio:', result_sr)
        Set_2 = np.max(Set2x, axis=0)
        self.Set_2 = Set_2 == 1
        Set_all = np.max(np.concatenate((Set1.reshape(1, -1), Set_2.reshape(1, -1))), axis=0)
        Set_all = Set_all == 1
        return [self.Set1, self.Set_2, Set_all, result_st, result_sr]

    def smc_deal_fun(self):
        if self.smc_deal[0]:
            # 使用Lasso
            if ~self.smc_deal[1]:
                result = LassoCV(cv=5).fit(self.x, self.y.reshape(-1)).coef_
                Set = result != 0
                if np.sum(Set) > 0:
                    self.x_name = self.x_name[Set]
                    self.x = self.x[:, Set]
                    print('Lasso筛选后指标列表为', self.x_name)
                else:
                    print('Lasso系数估计失败,不进行处理')
            # 使用双选择Lasso
            elif self.smc_deal[1]:
                ds_result = self.dslasso()
                if np.sum(ds_result[2]) > 0:
                    self.x_name = self.x_name[ds_result[2]]
                    self.x = self.x[:, ds_result[2]]
                    print('DS Lasso筛选后指标列表为', self.x_name)
                else:
                    print('DS Lasso系数估计失败,不进行处理')

    def __init__(self, data, smc_deal):
        '''
        :param data: pandas.DataFrame,第一列为目标变量,第二列及以后为指标
        :param smc_deal: 2个bool的list,例如[True,True],第一个代表是否对条件数大于1000的病态矩阵进行压缩估计,
        第二个代表是否用考虑潜变量的DS-Lasso
        1. 给出各个变量基本的描述性统计self.desc
        2. 画出各个变量的相关系数图self.corr_fig
        3. 取指标滞后一期的结果并于目标变量对齐
        4. 并计算指标的条件数self.cond,并根据是否大于1000提示不满秩的病态矩阵可能导致的强多重共线性问题
        4.1: 若self.cond不大于1000直接到5
        4.2: 若self.cond大于1000且smc_deal均为True,使用双选择Lasso筛选与y直接相关的变量集Set1和与Set1中变量相关的变量集Set2,
        即Set2通过间接影响Set1进而影响y,返回Set1和Set2的并集
        4.3: 若self.cond大于1000且smc_deal第一项True,第二项为False,是用Lasso进行变量筛选
        4.2和4.3使用cv=5的LassoCV实现,并要求筛选后的指标至少有一个,否则不处理
        5. 使用所有的指标和目标进行回归
        6. 对回归残差做平方进行ARCH的LM检验,若p小于0.1存在异方差问题则使用HC3重新回归对SE进行调整
        7. 检查除了常数项外指标的系数显著性t检验的p值是否小于0.1,否则剔除p值最大的指标
        8. 重复5-7至所有指标系数显著性t检验的p值均小于0.1或剔除了所有指标,并返回结果,幸存的指标就是有统计学上先行性的
        '''
        self.mem = data.iloc[:, 0].values

        self.data = data.iloc[:, 1:]
        self.smc_deal = smc_deal
        self.col_name = np.array(self.data.columns)

        self.x_name = self.col_name[1:]
        self.data_mat = self.data.values

        self.y = self.data_mat[1:, 0].reshape(-1, 1)
        self.x = self.data_mat[:-1, 1:]
        self.delete = True
        self.desc = self.data.describe()
        print(self.desc)
        '''
        self.corr_fig = plt.figure()
        self.corr_fig.add_subplot(
            sns.heatmap(np.corrcoef(self.data.transpose()), xticklabels=self.col_name, yticklabels=self.col_name))
        plt.title('目标变量和指标之间的相关系数矩阵')
        plt.show()
        '''
        self.cond = np.linalg.cond(self.x)

        if self.cond > 1000:
            print('条件数为:', self.cond, ',超过1000,可能有严重的多重共线性问题')
            self.smc_deal_fun()
        else:
            print('条件数为:', self.cond)

        while self.delete:
            self.mdl = OLS(self.y, add_constant(self.x)).fit()
            self.resid2 = self.mdl.resid ** 2
            self.ARCH_p = self.res2ARCHtest()
            if self.ARCH_p < 0.1:
                '''
                MacKinnon, White, 1985. Some heteroskedasticity consistent covariance matrix estimators with improved 
                finite sample properties, Journal of Econometrics
                
                About heteroskedasticity adjust SE:
                Jerry, 2011. Heteroskedasticity-Robust Inference in Finite Samples, NBER working paper
                '''
                self.mdl = OLS(self.y, add_constant(self.x)).fit(cov_type='HC3')
            self.pv = self.mdl.pvalues[1:]
            if (self.pv > 0.1).any():
                self.x_name = self.x_name[self.pv != np.max(self.pv)]
                self.x = self.x[:, self.pv != np.max(self.pv)]
            else:
                self.delete = False
        if len(self.x_name) != 0:
            print(self.mdl.summary())
            print('滞后性显著的指标列表为', self.x_name)
        else:
            print('模型没有估计出滞后性显著的列表')



# 测试数据的列数
cols_sim = 30

col_name = [(lambda x: 'x' + str(x))(x) for x in range(cols_sim + 1)]
col_name[0] = 'y'
df = pd.DataFrame(np.random.randint(10, 20, (1000, cols_sim + 1)), columns=col_name)
print(df.head(5))

result = model_fit(df, smc_deal=[True, True])
# 这就是原df中有先行性指标的列名
print(result.x_name)
print('finish')
