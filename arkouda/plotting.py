import numpy as np
from matplotlib import pyplot as plt

from arkouda.numeric import log


def log10(x):
    basechange = float(np.log10(np.exp(1)))
    return basechange*log(x)

def plot_dist(b, h, log=True, xlabel=None, newfig=True):
    if newfig: plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(b, h, marker='.', linestyle='solid')
    if log:
        plt.yscale('log')
    if xlabel is not None:
        plt.gca().set_xlabel(xlabel, fontsize=14)
    plt.gca().set_title('distribution')
    plt.subplot(1, 2, 2)
    plt.plot(b, np.cumsum(h)/np.sum(h), marker=None, linestyle='solid')
    plt.gca().set_ylim((0, 1))
    plt.gca().set_title('cumulative distribution')
    if xlabel is not None:
        plt.gca().set_xlabel(xlabel, fontsize=14)
