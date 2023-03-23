import numpy as np
import pandas as pd
import copy

df = pd.read_csv("wdbc.csv")

column_list = [column for column in df] #获得所有列名列表
num_col = copy.deepcopy(column_list)
del num_col[0:2]

minmax = (df[num_col] - df[num_col].min()) / (df[num_col].max() - df[num_col].min()) 
df_minmax = pd.concat([df[['ID','Diagnosis']],minmax],axis=1)

outputpath='C:/Users/shenxuan/Desktop/D1-minmax.csv'
df_minmax.to_csv(outputpath,sep=',',index=False,header=False)


