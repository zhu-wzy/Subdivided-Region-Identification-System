import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler

from factor_analyzer import FactorAnalyzer
from numpy.linalg import svd

###########################################################################################
# 显示所有列
pd.set_option('display.max_columns', None)
# 5000一列
pd.set_option('display.width', 5000)
# 显示所有行
# pd.set_option('display.max_rows', None)
# 对齐列名
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)


# ###############################################################################################
# 因子旋转 —— 方法2
# rotatefactors(am,'method','varimax')
def varimax(Phi, gamma=1, q=20, tol=1e-6):
    p, k = Phi.shape

    R = np.eye(k)

    d = 0

    for i in range(q):

        d_old = d

        Lambda = np.dot(Phi, R)

        u, s, vh = svd(
            np.dot(
                Phi.T,
                np.asarray(Lambda) ** 3 - (gamma / p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))
            )
        )

        R = np.dot(u, vh)

        d = np.sum(s)

        # print(d, d_old, d - d_old)

        if d - d_old < tol:
            break

    return np.dot(Phi, R)


# 主程序(因子旋转有2种方法实现，方法二的结果与matlab的rotatefactors函数最为接近)
def PCA_main(data, yuzhi=0.85, rotation=2):

    # ###################### 数据处理 #######################
    x = data.copy()

    # 标准化 不影响结果  归一化到均值为0
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    # %因子维度(行)
    n = x.shape[0]

    # 相关系数
    r = pd.DataFrame(x).corr()

    # ################## PCA主成分分析 ##########################
    #     % 利用相关系数矩阵进行主成分分析
    #     % vec1 的列是 r 的特向量,即主成分系数
    #     % val 为 r 的特征值
    #     % con1  为各个主成分的贡献率
    val, vec1 = np.linalg.eig(np.array(r))

    con1 = np.array([i / val.sum() for i in val]) * 100

    f1 = np.tile(np.sign(vec1.sum(axis=0)), (vec1.shape[0], 1))

    vec2 = vec1 * f1

    f2 = np.tile(np.sqrt(val), (vec2.shape[0], 1))

    a = vec2 * f2

    # ################## baoliu_chengfen ##########################
    baoliu_chengfen = yuzhi * 100

    # 保留前num个公因子
    print('累积贡献率：')
    for num in range(1, len(con1)):

        am = a[:, 0:num]

        # 累积贡献率
        con = con1[0:num].sum()
        print(num, con)

        if con >= baoliu_chengfen:
            break

        num = num + 1

    # ################## 因子旋转 ##########################
    # 方法1
    if rotation == 1:
        f = FactorAnalyzer(rotation='varimax', n_factors=num, method='principal')

        f.fit(am)

        bm = f.transform(am)

        t = f.loadings_

    else:
        bm = varimax(Phi=am, gamma=1, q=20)

    # % bt 前部分是旋转后的载荷矩阵  后部分为没有旋转的载荷矩阵
    bt = np.append(bm, a[:, num:], axis=1)

    # 计算贡献因子
    con2 = (bt ** 2).sum(axis=0)

    # 可以查看一下旋转后的贡献率
    check = [con1, con2 / sum(con2) * 100]

    # 旋转后的因子贡献率
    rate = con2 / sum(con2) * 100
    con2 = np.array([rate[0:i + 1].sum() for i in range(num)])

    # ################## 因子得分 ##########################
    # % 计算得分函数的系数 %得分函数系数公式： F=x R逆*载荷矩阵 (pinv还快一点)
    # coef = np.linalg.pinv(bm).T                         # 伪逆
    coef = np.linalg.lstsq(r, bm, rcond=None)[0]        # 最小二乘

    # 计算各个因子得分
    score = x.dot(coef)

    # 计算得分的权重
    weight = rate/sum(rate)

    explain = np.array([sum(rate[:i + 1]) for i in range(len(rate))])

    return score, explain, bm


# ###############################################################################################
if __name__ == '__main__':
    print('\n' + '--------------------------------------------------------------------')
    # 读表
    data = pd.read_excel(r'D:\work\一汽\运行数据分析\函数说明function myf_pca_data_processing_new()\数据.xlsx',
                         sheet_name='data', header=None)
    print('读表完成：' + time.strftime('%Y-%m-%d %H:%M:%S'))

    # 调用主函数（因子旋转使用方法2）
    PCA_data, explained, zaihejuzhen = PCA_main(data, yuzhi=0.85, rotation=2)

    print(PCA_data,
          '\n---------------------------------------------------\n',
          explained,
          '\n---------------------------------------------------\n',
          zaihejuzhen)
