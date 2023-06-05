import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# 加载数据集
iris_data = pd.read_csv('../dataset/iris_modified.csv')
cancer_data = pd.read_csv('../dataset/cancer_modified.csv')

# 选择目标属性
iris_target = 'petal_width'
cancer_target = 'Mean Radius'

# 对Iris数据集进行划分
iris_X = iris_data.drop('target', axis=1)
iris_y = iris_data[iris_target]

iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(iris_X, iris_y, test_size=0.2, random_state=42)

# 对Breast Cancer数据集进行划分
cancer_X = cancer_data.drop('Diagnosis', axis=1)
cancer_y = cancer_data[cancer_target]
cancer_X_train, cancer_X_test, cancer_y_train, cancer_y_test = train_test_split(cancer_X, cancer_y, test_size=0.2, random_state=42)

# 随机森林回归器
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(iris_X_train, iris_y_train)
rf_iris_predictions = rf_regressor.predict(iris_X_test)

rf_regressor.fit(cancer_X_train, cancer_y_train)
rf_cancer_predictions = rf_regressor.predict(cancer_X_test)

# AdaBoost回归器
adaboost_regressor = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(random_state=42), random_state=42)
adaboost_regressor.fit(iris_X_train, iris_y_train)
adaboost_iris_predictions = adaboost_regressor.predict(iris_X_test)

adaboost_regressor.fit(cancer_X_train, cancer_y_train)
adaboost_cancer_predictions = adaboost_regressor.predict(cancer_X_test)

# 随机森林性能评估
rf_iris_mse = mean_squared_error(iris_y_test, rf_iris_predictions)
rf_cancer_mse = mean_squared_error(cancer_y_test, rf_cancer_predictions)

# AdaBoost性能评估
adaboost_iris_mse = mean_squared_error(iris_y_test, adaboost_iris_predictions)
adaboost_cancer_mse = mean_squared_error(cancer_y_test, adaboost_cancer_predictions)

print("随机森林 - Iris数据集均方误差：", rf_iris_mse)
print("随机森林 - Breast Cancer数据集均方误差：", rf_cancer_mse)
print("AdaBoost - Iris数据集均方误差：", adaboost_iris_mse)
print("AdaBoost - Breast Cancer数据集均方误差：", adaboost_cancer_mse)