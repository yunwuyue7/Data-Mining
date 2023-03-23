import numpy as np
import pandas as pd
import copy


df = pd.read_csv("wdbc.csv")

column_list = [column for column in df] #获得所有列名列表
num_col = copy.deepcopy(column_list)
del num_col[0:2]

#使用NumPy计算Z得分
zscore = (df[num_col] - df[num_col].mean()) / df[num_col].std()
df_zscore = pd.concat([df[['ID','Diagnosis']],zscore],axis=1)

outputpath='C:/Users/shenxuan/Desktop/D1-zscore.csv'
df_zscore.to_csv(outputpath,sep=',',index=False,header=False)


