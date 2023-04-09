import numpy as np
import pandas as pd
import copy

df = pd.read_csv("wdbc.csv")
nums = [i for i in range(0, 32) if i > 1 ]
df_num = df.iloc[:,nums]

def decimal_scaling_norm(df):
    max_val = np.max(np.abs(df))
    scale = 0
    while max_val >= 1:
        scale += 1
        max_val /= 10
    return df / (10 ** scale)

# 对DataFrame进行Decimal Scaling标准化
decimal_scaled = df_num.apply(decimal_scaling_norm)
df_float = pd.concat([df[['ID','Diagnosis']],decimal_scaled],axis=1)

outputpath='C:/Users/shenxuan/Desktop/D1-float.csv'
df_float.to_csv(outputpath,sep=',',index=False,header=False)


