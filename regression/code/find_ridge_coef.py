import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("winequality-red.csv",sep=";")
df_minmax = (df - df.min()) / (df.max() - df.min()) 

# 分离特征和标签
X = df_minmax.drop('quality', axis=1)
y = df_minmax['quality']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建Ridge模型
ridge = Ridge()

# 定义正则化系数λ的范围
param_grid = {'alpha': np.logspace(-4, 4, 100)}

# 使用网格搜索选择最佳正则化系数
grid_search = GridSearchCV(ridge, param_grid, cv=5)
grid_search.fit(X_scaled, y)
best_alpha = grid_search.best_params_['alpha']

# 使用最佳正则化系数重新训练模型
ridge = Ridge(alpha=best_alpha)
ridge.fit(X_scaled, y)

# 获取正则化路径数据
alphas = np.logspace(-4, 4, 100)
coefs = []
for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_scaled, y)
    coefs.append(ridge.coef_)
coefs = np.array(coefs)

# 绘制正则化路径图
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(alphas, coefs)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.savefig('Ridge_coef.png', dpi=300)
plt.show()

# 确定稳定的超参数λ的取值
from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=5)
ridge_cv.fit(X_scaled, y)
print('Stable alpha:','%.3f'%ridge_cv.alpha_)