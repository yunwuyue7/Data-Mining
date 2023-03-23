import numpy as np
import pandas as pd
import copy
import math
import matplotlib.pyplot as plt

df = pd.read_csv("wdbc.csv")

column_list = [column for column in df] #获得所有列名列表
num_col = copy.deepcopy(column_list)
del num_col[0:2] 


# define the number of rows and columns in the subplot grid
nrows = 6
ncols = 5

# create a new figure object with a specific size
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))

# flatten the axs array to make it easier to iterate over
axs = axs.ravel()

# iterate over the continuous columns and create a boxplot for each one
for i, col in enumerate(num_col):
    # select the current subplot
    ax = axs[i]
    
    # create the boxplot on the current subplot
    ax.boxplot(df[col])
    
    # set the subplot title and axis labels
    ax.set_title('Box plot of column '+col)
    ax.set_xlabel('Data')
    ax.set_ylabel('Value')
    
    # hide the x-axis tick labels on all but the bottom row of subplots
    if i < (nrows - 1) * ncols:
        ax.set_xticklabels([])
    
    # hide the y-axis tick labels on all but the leftmost column of subplots
    if i % ncols != 0:
        ax.set_yticklabels([])
    
# adjust the layout of the subplots and display the figure
plt.tight_layout()

# save the figure to the current directory
fig.savefig('boxplots.png', dpi=300)

# show the figure
plt.show()