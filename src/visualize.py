import numpy as np
import matplotlib.pyplot as plt

def plot_W(W):
    row_indices, column_indices = np.indices(W.shape)
    plt.scatter(column_indices.flatten(), row_indices.flatten(),
                s=abs(W).flatten(), color='black', marker='s')
