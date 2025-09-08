"""
Plotting utilities for Arkouda data structures.

The `arkouda.plotting` module provides lightweight, matplotlib-based visualization
functions for Arkouda arrays and DataFrames. These tools are intended for exploratory
data analysis, especially for understanding distributions and skew across numeric or
categorical data columns.

Functions
---------
plot_dist(b, h, log=True, xlabel=None, newfig=True)
    Plot the histogram and cumulative distribution for binned data.
    Useful for visualizing data generated from `ak.histogram`.

hist_all(ak_df: DataFrame, cols: list = [])
    Generate histograms for all numeric columns in an Arkouda DataFrame
    (or a specified subset of columns). Automatically computes the number
    of bins using Doane’s formula and handles missing values, datetime,
    and categorical data appropriately.

Notes
-----
- These functions require `matplotlib.pyplot` and are meant for interactive
  Python sessions or Jupyter notebooks.
- `plot_dist` does not call `plt.show()` automatically; you must call it manually
  to display the plot.
- `hist_all` handles categorical grouping via Arkouda's `GroupBy` and supports
  `Datetime` and `Timedelta` plotting by converting to numeric types.

Examples
--------
>>> import arkouda as ak
>>> import numpy as np
>>> from arkouda.plotting import hist_all, plot_dist
>>> df = ak.DataFrame({'x': ak.array(np.random.randn(100))})
>>> hist_all(df)

>>> b, h = ak.histogram(ak.arange(10), 3)
>>> plot_dist(b.to_ndarray(), h[:-1].to_ndarray())
>>> import matplotlib.pyplot as plt
>>> plt.show()

See Also
--------
- matplotlib.pyplot
- arkouda.DataFrame
- arkouda.histogram

"""

import math
from typing import Optional

from matplotlib import pyplot as plt
import numpy as np

from arkouda.categorical import Categorical
from arkouda.dataframe import DataFrame
from arkouda.groupbyclass import GroupBy
from arkouda.numpy import histogram, isnan
from arkouda.numpy.pdarrayclass import pdarray, skew
from arkouda.numpy.pdarraycreation import arange
from arkouda.numpy.strings import Strings
from arkouda.numpy.timeclass import Datetime, Timedelta, date_range, timedelta_range

__all__ = [
    "hist_all",
    "plot_dist",
]


def plot_dist(b, h, log=True, xlabel=None, newfig=True):
    """
    Plot the distribution and cumulative distribution of histogram Data.

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
    >>> h = h[:-1]
    >>> ak.plot_dist(b.to_ndarray(), h.to_ndarray())

    Show the plot:
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


def hist_all(ak_df: DataFrame, cols: Optional[list[str]] = None):
    """
    Create a grid of histograms for numeric columns in an Arkouda DataFrame.

    Parameters
    ----------
    ak_df : DataFrame
        An Arkouda DataFrame containing the data to visualize.
    cols : list
        Optional. A list of column names to plot. If empty or not provided, all
        columns in the DataFrame are considered.

    Notes
    -----
    This function uses matplotlib to display a grid of histograms. It attempts to
    select a suitable number of bins using Doane's formula. Columns with
    non-numeric types will be grouped and encoded before plotting.

    Examples
    --------
    >>> import arkouda as ak
    >>> import numpy as np
    >>> from arkouda.plotting import hist_all
    >>> ak_df = ak.DataFrame({
    ...     "a": ak.array(np.random.randn(100)),
    ...     "b": ak.array(np.random.randn(100)),
    ...     "c": ak.array(np.random.randn(100)),
    ...     "d": ak.array(np.random.randn(100))
    ... })
    >>> hist_all(ak_df)

    """
    if not cols or len(cols) == 0:
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
            from typing import List

            cols_idx = cols.index
            if isinstance(cols_idx, List):
                ax = axes[cols_idx.index(col)]
            else:
                ax = axes[cols_idx(col)]
            x = ak_df[col]

            if x.dtype == "float64":
                x = x[~isnan(x)]

            n = len(x)
            g1 = skew(x)

        except ValueError:
            GB_df = GroupBy(ak_df[col])

            if not isinstance(GB_df.unique_keys, (Strings, Categorical, pdarray)):
                raise TypeError(
                    f"expected one of (Strings, Categorical, pdarray), "
                    f"got {type(GB_df.unique_keys).__name__!r}"
                )

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

        ax.bar(bins, h[1][:-1].to_ndarray(), width=bins[1] - bins[0])
        ax.set_title(col, size=8)
        if x.max() > 100 * x.min():
            ax.set_yscale("log")
