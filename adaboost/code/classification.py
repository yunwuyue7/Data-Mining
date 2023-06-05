import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def load_data():
    # 加载数据集
    iris_data = pd.read_csv('../dataset/iris_modified.csv')
    cancer_data = pd.read_csv('../dataset/cancer_modified.csv')
    return iris_data, cancer_data

def split_data(data, target_column, test_size):
    # 划分数据集
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def train_and_predict(X_train, X_test, y_train, y_test):
    # 随机森林分类器
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)
    rf_predictions = rf_classifier.predict(X_test)

    # AdaBoost分类器
    adaboost_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=42), random_state=42)
    adaboost_classifier.fit(X_train, y_train)
    adaboost_predictions = adaboost_classifier.predict(X_test)

    return rf_predictions, adaboost_predictions

def evaluate_accuracy(y_test, predictions):
    # 计算准确率
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

def evaluate_confusion_matrix(y_test, predictions):
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, predictions)
    return cm

def main():
    iris_data, cancer_data = load_data()

    test_sizes = [0.2, 0.4, 0.6]
    for test_size in test_sizes:
        print("Test Size:", test_size)

        iris_X_train, iris_X_test, iris_y_train, iris_y_test = split_data(iris_data, 'target', test_size)
        rf_iris_predictions, adaboost_iris_predictions = train_and_predict(iris_X_train, iris_X_test, iris_y_train, iris_y_test)

        iris_rf_accuracy = evaluate_accuracy(iris_y_test, rf_iris_predictions)
        iris_adaboost_accuracy = evaluate_accuracy(iris_y_test, adaboost_iris_predictions)

        iris_rf_cm = evaluate_confusion_matrix(iris_y_test, rf_iris_predictions)
        iris_adaboost_cm = evaluate_confusion_matrix(iris_y_test, adaboost_iris_predictions)

        print("随机森林 - Iris数据集准确率：", iris_rf_accuracy)
        print("随机森林 - Iris数据集混淆矩阵：")
        print(iris_rf_cm)
        print("AdaBoost - Iris数据集准确率：", iris_adaboost_accuracy)
        print("AdaBoost - Iris数据集混淆矩阵：")
        print(iris_adaboost_cm)

        cancer_X_train, cancer_X_test, cancer_y_train, cancer_y_test = split_data(cancer_data, 'Diagnosis', test_size)
        rf_cancer_predictions, adaboost_cancer_predictions = train_and_predict(cancer_X_train, cancer_X_test, cancer_y_train, cancer_y_test)

        cancer_rf_accuracy = evaluate_accuracy(cancer_y_test, rf_cancer_predictions)
        cancer_adaboost_accuracy = evaluate_accuracy(cancer_y_test, adaboost_cancer_predictions)

        cancer_rf_cm = evaluate_confusion_matrix(cancer_y_test, rf_cancer_predictions)
        cancer_adaboost_cm = evaluate_confusion_matrix(cancer_y_test, adaboost_cancer_predictions)

        print("随机森林 - Breast Cancer数据集准确率：", cancer_rf_accuracy)
        print("随机森林 - Breast Cancer数据集混淆矩阵：")
        print(cancer_rf_cm)
        print("AdaBoost - Breast Cancer数据集准确率：", cancer_adaboost_accuracy)
        print("AdaBoost - Breast Cancer数据集混淆矩阵：")
        print(cancer_adaboost_cm)
        print()

if __name__ == '__main__':
    main()
