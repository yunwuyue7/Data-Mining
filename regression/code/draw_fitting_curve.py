import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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

# Define the polynomial degrees to test
polynomial_degrees = [1, 2, 3, 4, 5]

# Plot the original data
plt.figure(figsize=(8, 6))
plt.plot(t, y, label='original data')

# Plot the data points from D1
plt.scatter(D1[:, 0], D1[:, 1], label='D1')

# Train and plot the models for each polynomial degree
for degree in polynomial_degrees:
    # Transform the input data to polynomial features
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(D1[:, 0].reshape(-1, 1))
    
    # Train the model with the training data
    model = LinearRegression()
    model.fit(X_train_poly, D1[:, 1].reshape(-1, 1))
    
    # Generate data points to plot the model curve
    t_plot = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    X_plot_poly = poly.transform(t_plot.reshape(-1, 1))
    y_plot = model.predict(X_plot_poly)
    
    # Plot the model curve
    plt.plot(t_plot, y_plot.flatten(), label=f'degree={degree}')
    
# Set the axis labels and title
plt.xlabel('t')
plt.ylabel('y')
plt.title('Fitting sine function with polynomial regression')

# Show the legend
plt.legend()

# save the fig
plt.savefig('poly-curve.png', dpi=300)

# Show the plot
plt.show()