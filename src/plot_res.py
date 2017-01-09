import numpy as np
import visualize
import matplotlib.pyplot as plt

if __name__ == '__main__':
    W_real = np.load("res/w_real.npy")
    W_ref = np.load("res/w_ref.npy")
    W_our = np.load("res/w_our.npy")

    plt.figure()

    plt.subplot(1,3,1)
    visualize.plot_W(W_real.T)

    plt.title("True W")
    plt.xlabel("K")
    plt.ylabel("D")

    plt.subplot(1,3,2)
    visualize.plot_W(W_ref.T)

    plt.title("Ref W")
    plt.xlabel("K")
    plt.ylabel("D")

    plt.subplot(1,3,3)
    visualize.plot_W(W_our.T)

    plt.title("Our W")
    plt.xlabel("K")
    plt.ylabel("D")

    bounds_ref = np.load("res/bounds_ref.npy")
    bounds_our = np.load("res/bounds_our.npy")

    plt.figure()

    plt.subplot(2,2,1)
    plt.plot(list(range(len(bounds_ref))), bounds_ref)

    plt.title("Bounds ref")
    plt.xlabel("Iter")
    plt.ylabel("Bound")

    plt.subplot(2,2,2)
    plt.plot(list(range(len(bounds_our))), bounds_our)

    plt.title("Bounds our")
    plt.xlabel("Iter")
    plt.ylabel("Bound")

    width = 300

    plt.subplot(2,2,3)
    plt.plot(list(range(len(bounds_ref))), bounds_ref)
    plt.ylim([bounds_ref[-1]-width, bounds_ref[-1]+width])

    plt.xlabel("Iter")
    plt.ylabel("Bound")

    plt.subplot(2,2,4)
    plt.plot(list(range(len(bounds_our))), bounds_our)
    plt.ylim([bounds_our[-1]-width, bounds_our[-1]+width])

    plt.xlabel("Iter")
    plt.ylabel("Bound")

    plt.show()
