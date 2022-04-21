import numpy as np  # type: ignore
from matplotlib import pyplot as plt  # type: ignore

def plot_dist(b, h, log=True, xlabel=None, newfig=True):
    """
    Plot the distribution and cumulative distribution of histogram Data

    Parameters
    ----------
    b : np.ndarray
        Bin edges
    h : np.ndarray
        Histogram data
    log : bool
        use log to scale y
    xlabel: str
        Label for the x axis of the graph
    newfig: bool
        Generate a new figure or not

    Notes
    -----
    This function does not return or display the plot. A user must have matplotlib imported in addition to arkouda to
    display plots. This could be updated to return the object or have a flag to show the resulting plots.
    See Examples Below.

    Examples
    --------
    >>> import arkouda as ak
    >>> from matplotlib import pyplot as plt
    >>> b, h = ak.histogram(ak.arange(10), 3)
    >>> ak.plot_dist(b, h.to_ndarray())
    >>> # to show the plot
    >>> plt.show()
    """
    if newfig:
        plt.figure(figsize=(12, 5))
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
