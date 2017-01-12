import numpy as np
import visualize
import matplotlib.pyplot as plt

filetype = "eps"
dpi = 1000
target = "plots/"
dims = (6, 2)

def save_plot(path):
    plt.savefig(target+path+"."+filetype, format=filetype, dpi=dpi)

def plot_save(W_true, W_comp, path, cutoff=None):
    plt.figure(figsize=dims)
    visualize.plot_W(visualize.sort_W(W_true, W_comp), threshmin=cutoff)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    save_plot(path)

def plot_bound(bounds, width, path):
    plt.figure()
    plt.plot(list(range(len(bounds))), bounds)
    plt.ylim([bounds[-1]-width, bounds[-1]+width])
    plt.xlabel("Iter")
    plt.ylabel("Bound")
    save_plot(path)

if __name__ == '__main__':
    # plot with threshold for clarity
    W_real = np.load("res/w_real.npy")
    W_our = np.load("res/w_our.npy")
    W_fa = np.load("res/w_fa.npy")
    W_ref = np.load("res/w_ref.npy")
    W_full = np.load("res/w_ref_full.npy")

    plot_save(W_real, W_real, "true")
    plot_save(W_real, W_our, "our")
    plot_save(W_real, W_fa, "fa")
    plot_save(W_real, W_ref, "ref")
    plot_save(W_real, W_full, "full")

    # plot zoomed in
    width = 300

    bounds_ref = np.load("res/bounds_ref.npy")
    bounds_our = np.load("res/bounds_our.npy")
    bounds_full = np.load("res/bounds_ref_full.npy")

    plot_bound(bounds_ref, width, "bounds_ref")
    plot_bound(bounds_our, width, "bounds_our")
    plot_bound(bounds_full, width, "bounds_full")
