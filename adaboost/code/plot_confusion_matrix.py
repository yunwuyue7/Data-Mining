import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'SimHei'

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

def plot_confusion_matrix(cm, labels, ax):
    # 绘制混淆矩阵图表
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)

def main():
    iris_data, cancer_data = load_data()

    test_sizes = [0.2, 0.4, 0.6]
    fig, axs = plt.subplots(nrows=2, ncols=len(test_sizes), figsize=(16, 8))

    for i, test_size in enumerate(test_sizes):
        ax1 = axs[0, i]
        ax2 = axs[1, i]

        iris_X_train, iris_X_test, iris_y_train, iris_y_test = split_data(iris_data, 'target', test_size)
        rf_iris_predictions, adaboost_iris_predictions = train_and_predict(iris_X_train, iris_X_test, iris_y_train, iris_y_test)

        iris_rf_accuracy = evaluate_accuracy(iris_y_test, rf_iris_predictions)
        iris_adaboost_accuracy = evaluate_accuracy(iris_y_test, adaboost_iris_predictions)

        iris_rf_cm = evaluate_confusion_matrix(iris_y_test, rf_iris_predictions)
        iris_adaboost_cm = evaluate_confusion_matrix(iris_y_test, adaboost_iris_predictions)

        plot_confusion_matrix(iris_rf_cm, labels=['setosa', 'versicolor', 'virginica'], ax=ax1)
        ax1.set_title("随机森林 - Iris数据集\n准确率: {:.2f}".format(iris_rf_accuracy), fontsize=12)
        plot_confusion_matrix(iris_adaboost_cm, labels=['setosa', 'versicolor', 'virginica'], ax=ax2)
        ax2.set_title("AdaBoost - Iris数据集\n准确率: {:.2f}".format(iris_adaboost_accuracy), fontsize=12)

        cancer_X_train, cancer_X_test, cancer_y_train, cancer_y_test = split_data(cancer_data, 'Diagnosis', test_size)
        rf_cancer_predictions, adaboost_cancer_predictions = train_and_predict(cancer_X_train, cancer_X_test, cancer_y_train, cancer_y_test)

        cancer_rf_accuracy = evaluate_accuracy(cancer_y_test, rf_cancer_predictions)
        cancer_adaboost_accuracy = evaluate_accuracy(cancer_y_test, adaboost_cancer_predictions)

        cancer_rf_cm = evaluate_confusion_matrix(cancer_y_test, rf_cancer_predictions)
        cancer_adaboost_cm = evaluate_confusion_matrix(cancer_y_test, adaboost_cancer_predictions)

        plot_confusion_matrix(cancer_rf_cm, labels=['M', 'B'], ax=axs[0, i])
        ax1.set_title("随机森林 - Breast Cancer数据集\n准确率: {:.2f}".format(cancer_rf_accuracy), fontsize=12)
        plot_confusion_matrix(cancer_adaboost_cm, labels=['M', 'B'], ax=axs[1, i])
        ax2.set_title("AdaBoost - Breast Cancer数据集\n准确率: {:.2f}".format(cancer_adaboost_accuracy), fontsize=12)

    plt.tight_layout()
    plt.savefig('../confusion_matrix.png')
    plt.show()
    
if __name__ == '__main__':
    main()
