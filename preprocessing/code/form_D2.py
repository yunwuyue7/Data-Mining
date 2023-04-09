import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

df = pd.read_csv("wdbc.csv")

df_d2 = copy.deepcopy(df[['radius_worst','texture_wost']])

noise = np.random.uniform(low=-40, high=40, size=(569, 2))
df_noisy = df_d2 + noise

outputpath='C:/Users/shenxuan/Desktop/D2.csv'
df_noisy.to_csv(outputpath,sep=',',index=False,header=False)