import numpy as np
import pandas as pd
import copy

df = pd.read_csv("wdbc.csv")

column_list = [column for column in df] #获得所有列名列表
num_col = copy.deepcopy(column_list)
del num_col[0:2] 

for i in num_col:
    print(i,'列的均值为%.5f'%df[i].mean(),'方差为%.5f'%df[i].std(),'\n')  