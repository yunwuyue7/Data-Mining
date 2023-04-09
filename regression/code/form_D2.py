import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("winequality-red.csv",sep=";")
df_minmax = (df - df.min()) / (df.max() - df.min()) 

df_d2 = df_minmax[['alcohol','volatile acidity','sulphates','quality']]
outputpath='C:/Users/shenxuan/Desktop/D2.csv'
df_d2.to_csv(outputpath,sep=',',index=False,header=False)