import numpy as np
import matplotlib.pyplot as plt

# 设置随机数种子，以便复现结果
np.random.seed(1234)

# 生成包含两个正弦周期的数据集
n_samples = 200
t = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
amplitude = 0.5
y = amplitude * np.sin(t) + amplitude * np.sin(2 * t)

# 在数据集中添加随机扰动
noise = np.random.randn(n_samples) * 0.1
y += noise

# 对数据集进行均匀采样，生成数据集 D1
n_samples_D1 = 20
indices = np.linspace(0, n_samples-1, n_samples_D1, dtype=np.int32)
D1 = np.zeros((n_samples_D1, 2))
D1[:, 0] = t[indices]
D1[:, 1] = y[indices]

# 可视化数据集
plt.plot(t, y, label='original data')
plt.scatter(D1[:, 0], D1[:, 1], label='D1')
plt.legend()
plt.savefig('D1_visual.png', dpi=300)
plt.show()

np.savetxt('data.csv', D1, delimiter=',', header='t,y', comments='')