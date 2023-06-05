import pandas as pd
from sklearn.datasets import load_iris

def load_and_save_iris_data():
    # 加载 IRIS 数据集
    iris = load_iris()

    # 创建 DataFrame 并设置列名
    iris_data = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    iris_data['target'] = iris.target

    # 添加索引作为列名
    iris_data.columns.name = 'index'

    # 保存为csv文件
    iris_data.to_csv('../dataset/iris_modified.csv', index=False)

def load_and_save_cancer_data():
    data = pd.read_csv('../dataset/wdbc.csv')

    column_names = ['ID number','Diagnosis',
                    'Mean Radius', 'Mean Texture', 'Mean Perimeter','Mean Area', 'Mean Smoothness', 
                    'Mean Compactness','Mean Concavity','Mean Concave points', 'Mean Symmetry', 'Mean Fractal dimension',
                    'Radius SE', 'Texture SE', 'Perimeter SE','Area SE', 'Smoothness SE', 
                    'Compactness SE','Concavity SE','Concave points SE', 'Symmetry SE', 'Fractal dimension SE',
                    'Worst Radius', 'Worst Texture', 'Worst Perimeter','Worst Area', 'Worst Smoothness', 
                    'Worst Compactness','Worst Concavity','Worst Concave points', 'Worst Symmetry', 'Worst Fractal dimension']

    data.columns = column_names
    data.columns.name = 'index'

    data.to_csv('../dataset/cancer_modified.csv', index=False)

def main():
    load_and_save_iris_data()
    load_and_save_cancer_data()

if __name__ == '__main__':
    main()
