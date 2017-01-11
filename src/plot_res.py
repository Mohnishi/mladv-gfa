import numpy as np
import visualize
import matplotlib.pyplot as plt

filetype = "eps"
dpi = 1000
target = "plots/"
dims = (2, 6)
factor = 5000

if __name__ == '__main__':
    # plot with threshold for clarity
    W_real = np.load("res/w_real.npy")
    W_our = np.load("res/w_our.npy")
    W_ref = np.load("res/w_ref.npy")
    W_full = np.load("res/w_ref_full.npy")

    plt.figure(figsize=dims)

    visualize.plot_W(W_real.T)

    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    plt.savefig(target+"true.eps", format=filetype, dpi=dpi)

    plt.figure(figsize=dims)

    cutoff = np.amin(np.abs(W_our)) * factor
    visualize.plot_W(visualize.sort_W(W_real, W_our).T, threshmin=cutoff)

    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    plt.savefig(target+"our.eps", format=filetype, dpi=dpi)

    plt.figure(figsize=dims)
    cutoff = np.amin(np.abs(W_ref)) * factor
    visualize.plot_W(visualize.sort_W(W_real, W_ref).T, threshmin=cutoff)

    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    plt.savefig(target+"ref.eps", format=filetype, dpi=dpi)

    plt.figure(figsize=dims)
    cutoff = np.amin(np.abs(W_full)) * factor
    visualize.plot_W(visualize.sort_W(W_real, W_full).T, threshmin=cutoff)

    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    plt.savefig(target+"full.eps", format=filetype, dpi=dpi)

    bounds_ref = np.load("res/bounds_ref.npy")
    bounds_our = np.load("res/bounds_our.npy")
    bounds_full = np.load("res/bounds_ref_full.npy")

    # plot zoomed in
    width = 300

    plt.figure()
    plt.plot(list(range(len(bounds_ref))), bounds_ref)
    plt.ylim([bounds_ref[-1]-width, bounds_ref[-1]+width])
    plt.xlabel("Iter")
    plt.ylabel("Bound")
    plt.savefig(target+"bound_ref.eps", format=filetype, dpi=dpi)

    plt.figure()
    plt.plot(list(range(len(bounds_our))), bounds_our)
    plt.ylim([bounds_our[-1]-width, bounds_our[-1]+width])
    plt.xlabel("Iter")
    plt.ylabel("Bound")
    plt.savefig(target+"bound_our.eps", format=filetype, dpi=dpi)

    plt.figure()
    plt.plot(list(range(len(bounds_full))), bounds_full)
    plt.ylim([bounds_full[-1]-width, bounds_full[-1]+width])
    plt.xlabel("Iter")
    plt.ylabel("Bound")
    plt.savefig(target+"bound_full.eps", format=filetype, dpi=dpi)
