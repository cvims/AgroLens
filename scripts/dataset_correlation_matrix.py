#!/usr/bin/env python
# Script to read a data table, calculate the correlation matrix and plot it

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def compute_and_plot_correlation(csv_file, substring="", exclude=False):
    # Read the CSV file using pandas
    data = pd.read_csv(csv_file)

    # choose only data where the name contains a substring or not contains a substring,
    # depending on the parameter 'exclude'
    if exclude == False:
        filtered_columns = [col for col in data.columns if substring in col]
        filtered_data = data[filtered_columns]
    else:
        filtered_columns = [col for col in data.columns if substring not in col]
        filtered_data = data[filtered_columns]

    # calculate the correlation coefficent matrix for all choosen data
    numeric_data = filtered_data.select_dtypes(include=[np.number])
    corr_matrix = np.corrcoef(numeric_data.T)

    # plot the matrix
    plt.ion()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        xticklabels=numeric_data.columns,
        yticklabels=numeric_data.columns,
    )
    plt.title("Correlation Matrix", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()


"""
This Part must be adjusted by the user
"""

# Add here your data location (have to be a .csv)
csv_file = "../Data/Datasets/Model_A_Dataset_v6_2025-01-05.csv"

# uncomment this section if you want to plot the correlation matrix for all data columns
compute_and_plot_correlation(csv_file)

# uncomment this section if you only want specific data columns that have the substring in their name
# substring = 'normalized'
# compute_and_plot_correlation(csv_file,substring)


# uncomment this section if you only want specific data columns that have NOT the substring in their name
# substring = 'normalized'
# compute_and_plot_correlation(csv_file,substring,True)


# uncomment if you dont want to save the figure
plt.savefig("Dataset-A_all_Correlation-Plot.eps", format="eps")
# plt.savefig('Dataset-A_all_Correlation-Plot.eps', format='png')
