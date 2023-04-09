import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("wdbc.csv")

def LOF(X, k):
    """
    计算数据集X的局部离群因子(LOF)值。

    参数：
    X: 二维数组，表示数据集。
    k: int，表示k-邻域的大小。

    返回：
    lof_scores: 一维数组，表示每个数据点的LOF值。
    """

    # 计算每个数据点到其他数据点的距离
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, indices = nbrs.kneighbors(X)

    # 计算每个数据点的k-邻域可达距离
    reachability_dists = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        k_dist = distances[i, -1] # k-邻域距离
        reachability_dists[i] = max(k_dist, np.sum(distances[indices[i], -1]) / k)

    # 计算每个数据点的局部可达密度(LRD)
    lrd = 1.0 / (np.mean(reachability_dists[indices], axis=1) + 1e-10)

    # 计算每个数据点的LOF值
    lof_scores = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        lof_scores[i] = np.mean(lrd[indices[i]]) / lrd[i]

    return lof_scores

df_d2 = copy.deepcopy(df[['radius_worst','texture_wost']])
noise = np.random.uniform(low=-40, high=40, size=(569, 2))
df_noisy = df_d2 + noise

# 使用LOF算法计算LOF值
lof_scores = LOF(df_noisy.values, k=5)

# 标识离群点
thresholds = [1, 1.1, 1.5]
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,6))
for i, threshold in enumerate(thresholds):
    lof_scores = LOF(df_noisy.values, k=5)
    outliers = lof_scores > threshold
    
    # 绘制散点图
    plt.subplot(1, 3, i+1)
    plt.scatter(df_noisy.loc[~outliers, 'radius_worst'], df_noisy.loc[~outliers, 'texture_wost'], color='blue', label='normal')
    plt.scatter(df_noisy.loc[outliers, 'radius_worst'], df_noisy.loc[outliers, 'texture_wost'], color='red', label='outlier')
    plt.legend()
    plt.title(f"Threshold={threshold}")
    
plt.tight_layout()
fig.savefig('scatterplot.png', dpi=300)
plt.show()


