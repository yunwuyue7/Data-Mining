import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_data():
    # 加载数据集
    cancer_data = pd.read_csv('../dataset/cancer_modified.csv')
    return cancer_data

def split_data(data, target_column, test_size):
    # 划分数据集为训练集、测试集和验证集
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

def stacking_model(base_classifiers, meta_classifier, X_train, y_train, X_val, y_val, X_test):
    # 创建Stacking模型
    base_classifier_predictions = np.empty((X_val.shape[0], len(base_classifiers)))

    for i, classifier in enumerate(base_classifiers):
        classifier.fit(X_train, y_train)
        base_classifier_predictions[:, i] = classifier.predict(X_val)

    meta_classifier.fit(base_classifier_predictions, y_val)

    base_classifier_predictions_test = np.empty((X_test.shape[0], len(base_classifiers)))

    for i, classifier in enumerate(base_classifiers):
        base_classifier_predictions_test[:, i] = classifier.predict(X_test)

    stacked_predictions = meta_classifier.predict(base_classifier_predictions_test)

    return stacked_predictions

def main():
    cancer_data = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(cancer_data, 'Diagnosis', test_size=0.2)

    base_classifiers = [
        RandomForestClassifier(random_state=42),
        GradientBoostingClassifier(random_state=42),
        AdaBoostClassifier(random_state=42)
    ]
    meta_classifier = LogisticRegression()

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)

    stacked_predictions = stacking_model(base_classifiers, meta_classifier, X_train, y_train, X_val, y_val, X_test)

    accuracy = accuracy_score(y_test, stacked_predictions)
    print("Stacking模型在测试集上的准确率：", accuracy)

if __name__ == '__main__':
    main()
