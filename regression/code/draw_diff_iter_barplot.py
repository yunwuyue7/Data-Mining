import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("winequality-red.csv",sep=";")
df_minmax = (df - df.min()) / (df.max() - df.min()) 
df_d2 = df_minmax[['alcohol','volatile acidity','sulphates','quality']]

# 定义超参数和循环次数
alpha = 37.649
n_iter_list = [1, 2, 3, 4, 5]
mae_list = []
rmse_list = []

for n_iter in n_iter_list:    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(df_d2[['alcohol', 'volatile acidity', 'sulphates']], df_d2['quality'], test_size=0.2)

    # 定义Ridge回归模型
    ridge = Ridge(alpha=alpha)

    # 训练模型
    for i in range(n_iter):
        ridge.fit(X_train, y_train)

    # 预测结果
    y_pred = ridge.predict(X_test)

    # 计算MAE和RMSE
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # 存储结果
    mae_list.append(mae)
    rmse_list.append(rmse)

fig, ax = plt.subplots(figsize=(8, 6))
bar_width = 0.35
opacity = 0.8
index = np.arange(len(n_iter_list))
ax.bar(index, mae_list, bar_width, alpha=opacity, color='b', label='MAE')
ax.bar(index + bar_width, rmse_list, bar_width, alpha=opacity, color='g', label='RMSE')
ax.set_xlabel('n_iter')
ax.set_ylabel('Error')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(n_iter_list)
ax.legend()
plt.savefig('diff_iter_MSE_RMSE.png', dpi=300)
plt.show()
