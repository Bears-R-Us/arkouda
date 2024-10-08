import math

import numpy as np
from matplotlib import pyplot as plt

from arkouda.dataframe import DataFrame
from arkouda.groupbyclass import GroupBy
from arkouda.numpy import histogram, isnan
from arkouda.pdarrayclass import skew
from arkouda.pdarraycreation import arange
from arkouda.timeclass import Datetime, Timedelta, date_range, timedelta_range


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
    This function does not return or display the plot. A user must have matplotlib imported in
    addition to arkouda to display plots. This could be updated to return the object or have a
    flag to show the resulting plots.
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
    plt.plot(b, h, marker=".", linestyle="solid")
    if log:
        plt.yscale("log")
    if xlabel is not None:
        plt.gca().set_xlabel(xlabel, fontsize=14)
    plt.gca().set_title("distribution")
    plt.subplot(1, 2, 2)
    plt.plot(b, np.cumsum(h) / np.sum(h), marker=None, linestyle="solid")
    plt.gca().set_ylim((0, 1))
    plt.gca().set_title("cumulative distribution")
    if xlabel is not None:
        plt.gca().set_xlabel(xlabel, fontsize=14)


def hist_all(ak_df: DataFrame, cols: list = []):
    """
    Create a grid plot histogramming all numeric columns in ak dataframe

    Parameters
    ----------
    ak_df : ak.DataFrame
        Full Arkouda DataFrame containing data to be visualized
    cols : list
        (Optional) A specified list of columns to be plotted

    Notes
    -----
    This function displays the plot.

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda.plotting import hist_all
    >>> ak_df = ak.DataFrame({"a": ak.array(np.random.randn(100)),
                              "b": ak.array(np.random.randn(100)),
                              "c": ak.array(np.random.randn(100)),
                              "d": ak.array(np.random.randn(100))
                              })
    >>> hist_all(ak_df)
    """

    if len(cols) == 0:
        cols = ak_df.columns

    num_rows = int(math.ceil(len(cols) ** 0.5))
    num_cols = (len(cols) + num_rows - 1) // num_rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    fig.tight_layout(pad=2.0)

    if isinstance(axes, plt.Axes):
        axes = np.array(axes).flatten()
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for col in cols:
        try:
            ax = axes[cols.index(col)]
            x = ak_df[col]

            if x.dtype == "float64":
                x = x[~isnan(x)]

            n = len(x)
            g1 = skew(x)

        except ValueError:
            GB_df = GroupBy(ak_df[col])
            new_labels = arange(GB_df.unique_keys.size)
            newcol = GB_df.broadcast(new_labels)
            x = newcol[: ak_df.size]

            if x.dtype == "float64":
                x = x[~isnan(x)]

            n = len(x)
            g1 = skew(x)

        sigma_g1 = math.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
        # Doane's Formula
        num_bins = int(1 + math.log2(n) + math.log2(1 + abs(g1) / sigma_g1))

        # Compute histogram counts in arkouda
        h = histogram(x, num_bins)
        # Compute bins in numpy
        if isinstance(x, Datetime):
            # Matplotlib has trouble plotting np.datetime64 and np.timedelta64
            bins = date_range(x.min(), x.max(), periods=num_bins).to_ndarray().astype("int")
        elif isinstance(x, Timedelta):
            bins = timedelta_range(x.min(), x.max(), periods=num_bins).to_ndarray().astype("int")
        else:
            bins = np.linspace(x.min(), x.max(), num_bins + 1)[:-1]

        ax.bar(bins, h[1].to_ndarray(), width=bins[1] - bins[0])
        ax.set_title(col, size=8)
        if x.max() > 100 * x.min():
            ax.set_yscale("log")
