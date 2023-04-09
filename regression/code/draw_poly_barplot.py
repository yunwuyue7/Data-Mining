import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

# Define the values of m to test
m_values = [1, 2, 3, 4, 5]

# Split the dataset into training and test sets
X = D1[:, 0].reshape(-1, 1)
y = D1[:, 1].reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Define arrays to store the MAE and RMSE for each value of m
mae_values = []
rmse_values = []

# Train and test the model for each value of m
for m in m_values:
    # Transform the input data to polynomial features
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(m)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Train the model with the training data
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Test the model with the test data
    y_pred = model.predict(X_test_poly)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Append the MAE and RMSE to the arrays
    mae_values.append(mae)
    rmse_values.append(rmse)

# Plot the results
fig, ax = plt.subplots(figsize=(8, 6))
bar_width = 0.35
opacity = 0.8
index = np.arange(len(m_values))
ax.bar(index, mae_values, bar_width, alpha=opacity, color='b', label='MAE')
ax.bar(index + bar_width, rmse_values, bar_width, alpha=opacity, color='g', label='RMSE')
ax.set_xlabel('m')
ax.set_ylabel('Error')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(m_values)
ax.legend()
plt.savefig('poly-barplot.png', dpi=300)
plt.show()






