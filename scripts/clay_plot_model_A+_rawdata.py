#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
plot the raw pixel data for virualization and interpretation
"""

filepath = "../Model_A+_Soil+Sentinel_9x9.csv"


data_header = pd.read_csv(filepath, sep=",", nrows=0)
print(data_header.shape)
idx_start = data_header.columns.get_loc("B01_1_normalized")
idx_end = data_header.columns.get_loc("B12_81_normalized")
print(idx_start, idx_end)

data_bands = pd.read_csv(
    filepath, sep=",", usecols=range(idx_start, idx_end + 1), header=0
)
# print(data_long.head, data_bands.head)
print(data_bands.shape)


data_bands_np = data_bands.to_numpy()
data_bands_clay = np.zeros((data_bands_np.shape[0], 12, 9, 9))
for i in range(0, data_bands_np.shape[0]):
    for j in range(0, 12):
        data_bands_clay[i, j, :, :] = data_bands_np[
            i, (0 + 81 * j) : (81 + 81 * j)
        ].reshape(9, 9)
print(data_bands_clay[0:1], data_bands_clay.shape)


m = 8  # grid size of the plot -> eg. 8x8

fig, axs = plt.subplots(m, m, figsize=(20, 20))
for i in range(0, 12):  # 12 bands pro data set
    for idx, ax in enumerate(axs.flatten()):
        ax.imshow(data_bands_clay[idx, i], cmap="bwr")
        ax.set_axis_off()
        ax.set_title(idx)
    plt.tight_layout()
    plt.savefig(f"plot_model_A+_rawdata_band{i}.png", dpi=300)
# plt.show()
