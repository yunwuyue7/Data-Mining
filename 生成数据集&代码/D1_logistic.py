import numpy as np
import pandas as pd
import copy

df = pd.read_csv("wdbc.csv")
nums = [i for i in range(0, 32) if i > 1 ]
df_num = df.iloc[:,nums]

def log_norm(df):
    return 1 / (1 + np.exp(-df))

log = log_norm(df_num)
df_log = pd.concat([df[['ID','Diagnosis']],log],axis=1)

outputpath='C:/Users/shenxuan/Desktop/D1-log.csv'
df_log.to_csv(outputpath,sep=',',index=False,header=False)