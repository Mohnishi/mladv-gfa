import numpy as np
import visualize
import matplotlib.pyplot as plt

if __name__ == '__main__':
    W_real = np.load("res/w_real.npy")
    W_ref = np.load("res/w_ref.npy")
    W_our = np.load("res/w_our.npy")

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

    plt.show()
