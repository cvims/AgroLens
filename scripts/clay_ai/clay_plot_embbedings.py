#!/usr/bin/env python
# Script to plot generated Clay embeddings loaded from a CSV table
import matplotlib.pyplot as plt
import numpy as np

"""
1024 embeddings are created for every data set.
1024 = 32x32
It means that thr 1024 embeddings can be visualized as a plot of 32x32 pixels
"""

filepath_embeddings = "250116_clay_embeddings.csv"


data = np.loadtxt(filepath_embeddings, delimiter=",")

m = 4  # grid size of the plot
n = (
    m * m
)  # number of subplots = the number of data sets, of which the embeddings are plotted
data_32x32 = np.zeros((n, 32, 32))
for i in range(0, n):
    data_32x32[i, :] = data[i].reshape(32, 32)


fig, axs = plt.subplots(m, m, figsize=(20, 20))
for idx, ax in enumerate(axs.flatten()):
    ax.imshow(data_32x32[idx], cmap="bwr")
    ax.set_axis_off()
    ax.set_title(idx)
plt.tight_layout()
plt.savefig(f"250116_clay_embeddings_plot_{m}x{m}.png", dpi=300)
plt.show()
