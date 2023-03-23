import numpy as np
import pandas as pd
import copy

df = pd.read_csv("wdbc.csv")

column_list = [column for column in df] #获得所有列名列表
num_col = copy.deepcopy(column_list)
del num_col[0:2] 


def entropy(y):
    # 计算熵
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.sum(p * np.log2(p))

def information_gain(x, y, split):
    # 计算分割点为split时的信息增益
    left_mask = x <= split
    right_mask = x > split
    left_entropy = entropy(y[left_mask])
    right_entropy = entropy(y[right_mask])
    p_left = np.sum(left_mask) / len(x)
    p_right = np.sum(right_mask) / len(x)
    return entropy(y) - (p_left * left_entropy + p_right * right_entropy)

def discretize(x, y, num_bins):
    # 将x离散化为num_bins个区间
    x_min, x_max = x.min(), x.max()
    bin_width = (x_max - x_min) / num_bins
    splits = [x_min + i * bin_width for i in range(1, num_bins)]
    gains = [information_gain(x, y, split) for split in splits]
    best_split = splits[np.argmax(gains)]
    # 将x离散化为{<=best_split, >best_split}两个值
    return np.where(x <= best_split, 0, 1)

def CAIM(df, col, num_bins):
    # 将数据按照指定列排序
    data = df.sort_values(by=col)[[col, 'Diagnosis']]
    data.reset_index(inplace=True, drop=True)
    n = len(data)

    # 生成候选划分点
    cut_points = [data[col].iloc[i] for i in range(num_bins - 1)]
    max_cramer_v = 0
    final_cut_point = None

    # 计算每个候选划分点的Cramer's V值
    for cut_point in cut_points:
        left = data[data[col] <= cut_point]
        right = data[data[col] > cut_point]
        n_left = len(left)
        n_right = len(right)

        # 计算左侧和右侧的类别属性分布
        left_dist = left['Diagnosis'].value_counts().sort_index().tolist()
        right_dist = right['Diagnosis'].value_counts().sort_index().tolist()

        # 计算左侧和右侧的总类别属性数量
        n_total_left = sum(left_dist)
        n_total_right = sum(right_dist)

        # 计算左侧和右侧的类别属性概率分布
        left_prob = [x / n_total_left for x in left_dist]
        right_prob = [x / n_total_right for x in right_dist]

        # 计算Cramer's V值
        p_total = (n_total_left + n_total_right) / n
        p_left = n_total_left / n
        p_right = n_total_right / n
        v = p_total * (
            sum([x**2 for x in left_prob]) / p_left +
            sum([x**2 for x in right_prob]) / p_right -
            (sum([x**2 for x in left_prob + right_prob]) / p_total)
        )

        # 更新最大Cramer's V值和最终的划分点
        if v > max_cramer_v:
            max_cramer_v = v
            final_cut_point = cut_point

    # 返回最终的划分点
    return final_cut_point


#等距离散化
df_radius_mean_series = copy.deepcopy(df['radius_mean'])
df_radius_mean = df_radius_mean_series.to_frame()
df_radius_mean['bins'] = pd.cut(df_radius_mean['radius_mean'], bins=5)
df_radius_mean_discrete = pd.DataFrame({'radius_mean': df_radius_mean['bins']})

#信息增益离散化
tmp = discretize(df['area_mean'], df['Diagnosis'], 5)
df_area_mean_discrete = pd.DataFrame({'area_mean': tmp})

#卡方离散化
df_texture_mean_series = copy.deepcopy(df['texture_mean'])
df_texture_mean, bins = pd.qcut(df_texture_mean_series, q=4, retbins=True, duplicates='drop', labels=False)
df_texture_mean_discrete = df_texture_mean.to_frame()

#CAIM离散化
cut_point = CAIM(df, 'perimeter_mean', 5)
df['perimeter_mean_discrete'] = pd.cut(df['perimeter_mean'], [-np.inf, cut_point, np.inf], labels=[0, 1])
df.groupby('perimeter_mean_discrete')['Diagnosis'].value_counts(normalize=True)
df_perimeter_mean_discrete = pd.DataFrame({'perimeter_mean': df['perimeter_mean_discrete']}) 

#用concat函数四种离散化方法处理的四列数据与其他数据拼接，形成D1_discrete
df_discrete = pd.concat([df[['ID','Diagnosis']],df_radius_mean_discrete,df_texture_mean_discrete,df_perimeter_mean_discrete,df_area_mean_discrete,df.iloc[0:,6:]],axis=1)

#指定路径输出
outputpath='C:/Users/shenxuan/Desktop/D1-discrete.csv'
df_discrete.to_csv(outputpath,sep=',',index=False,header=False)
