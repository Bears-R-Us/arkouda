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

Save the figure to disk:
>>> fig, axes = hist_all(df)
>>> fig.savefig("hist_all.png")
>>> b, h = ak.histogram(ak.arange(10), 3)
>>> plot_dist(b.to_ndarray(), h[:-1].to_ndarray())
(<Figure size 1200x500 with 2 Axes>, array([<Axes: title={'center': 'distribution'}>,
       <Axes: title={'center': 'cumulative distribution'}>], dtype=object))
>>> import matplotlib.pyplot as plt
>>> plt.show()

See Also
--------
- matplotlib.pyplot
- arkouda.DataFrame
- arkouda.histogram

"""

from __future__ import annotations

import math

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.figure import Figure
from numpy.typing import NDArray

import arkouda as ak

from arkouda.categorical import Categorical
from arkouda.dataframe import DataFrame
from arkouda.groupbyclass import GroupBy
from arkouda.numpy import histogram, isnan
from arkouda.numpy.pdarrayclass import skew
from arkouda.numpy.pdarraycreation import arange
from arkouda.numpy.strings import Strings
from arkouda.numpy.timeclass import Datetime, Timedelta, date_range, timedelta_range
from arkouda.pdarrayclass import pdarray


__all__ = [
    "hist_all",
    "plot_dist",
]


def plot_dist(
    b: pdarray | NDArray[np.floating],
    h: pdarray | NDArray[np.floating],
    *,
    log: bool = True,
    xlabel: Optional[str] = None,
    newfig: bool = True,
    show: bool = False,
) -> Tuple[Figure, np.ndarray]:
    """
    Plot the distribution and cumulative distribution of histogram data.

    Parameters
    ----------
    b : arkouda.pdarray or numpy.ndarray
        Histogram bin edges (length N+1) or bin centers (length N).
    h : arkouda.pdarray or numpy.ndarray
        Histogram counts. Accepts length N or N+1 (Arkouda-like extra last bin).
    log : bool, default True
        If True, use a log scale for the y-axis of the distribution plot.
    xlabel : str, optional
        Label for the x-axis.
    newfig : bool, default True
        If True, create a new figure; otherwise draw into the current figure.
    show : bool, default False
        If True, call ``plt.show()`` before returning.

    Returns
    -------
    tuple[matplotlib.figure.Figure, numpy.ndarray]
        (fig, axes) where axes[0] is the distribution, axes[1] the cumulative.

    Notes
    -----
    If ``h`` is one element longer than expected (as with ``ak.histogram``),
    the final element is dropped automatically.

    Examples
    --------
    Using Arkouda's histogram:
    >>> import arkouda as ak
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> from arkouda.plotting import plot_dist
    >>> edges, counts = ak.histogram(ak.arange(10), 3)
    >>> fig, axes = plot_dist(edges, counts)
    >>> fig.savefig("dist.png")

    Using NumPy's histogram:
    >>> data = np.random.randn(1000)
    >>> counts, edges = np.histogram(data, bins=20)
    >>> fig, axes = plot_dist(edges, counts, xlabel="Value")
    >>> plt.show()

    """

    def to_ndarray(arr: pdarray | NDArray[np.floating]) -> NDArray[np.floating]:
        if isinstance(arr, pdarray):
            nbytes = arr.nbytes
            if nbytes > ak.client.maxTransferBytes:
                raise ValueError(
                    f"Array too large to transfer: {nbytes} bytes (max {ak.client.maxTransferBytes})"
                )
            return arr.to_ndarray()
        return np.asarray(arr)

    b = to_ndarray(b).astype(np.float64, copy=False)
    h = to_ndarray(h).astype(np.float64, copy=False)

    if b.ndim != 1 or h.ndim != 1:
        raise ValueError("b and h must be 1-D arrays.")

    # Normalize Arkouda-style extra last count
    if b.size == h.size + 1:
        # edges (N+1) + counts (N)
        x = 0.5 * (b[:-1] + b[1:])
    elif b.size == h.size - 1:
        # centers (N) + counts (N+1) -> drop last count, use centers
        h = h[:-1]
        x = b.astype(np.float64, copy=False)
    elif b.size == h.size:
        # centers (N) + counts (N)
        x = b.astype(np.float64, copy=False)
    else:
        raise ValueError(
            f"Length mismatch: len(b)={b.size} vs len(h)={h.size}. "
            "Expected: edges (N+1) with counts N or N+1; or centers N with counts N or N+1."
        )

    if newfig:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig = plt.gcf()
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        axes = np.array([ax1, ax2], dtype=object)

    # Distribution
    ax = axes[0]
    ax.plot(x, h, marker=".", linestyle="solid")
    if log and np.any(h > 0):
        ax.set_yscale("log")
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title("distribution")

    # Cumulative (normalized)
    ax = axes[1]
    total = float(np.sum(h))
    cdf = np.cumsum(h / total) if total > 0 else np.zeros_like(h, dtype=float)
    ax.plot(x, cdf, linestyle="solid")
    ax.set_ylim((0, 1))
    ax.set_title("cumulative distribution")
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=12)

    if show:
        plt.show()

    return fig, axes


def hist_all(ak_df: DataFrame, cols: Optional[list[str]] = None):
    """
    Create a grid of histograms for numeric columns in an Arkouda DataFrame.

    Parameters
    ----------
    ak_df : DataFrame
        An Arkouda DataFrame containing the data to visualize.
    cols : list, optional
        A list of column names to plot. If empty or not provided, all
        columns in the DataFrame are considered.

    Returns
    -------
    tuple[matplotlib.figure.Figure, numpy.ndarray]
        A tuple containing the matplotlib Figure and an array of Axes objects.

    Notes
    -----
    This function uses matplotlib to display a grid of histograms. It attempts to
    select a suitable number of bins using Doane's formula. Columns with
    non-numeric types will be grouped and encoded before plotting.

    Examples
    --------
    Basic usage with all columns:
    >>> import arkouda as ak
    >>> import numpy as np
    >>> from arkouda.plotting import hist_all
    >>> ak_df = ak.DataFrame({
    ...     "a": ak.array(np.random.randn(100)),
    ...     "b": ak.array(np.random.randn(100)),
    ...     "c": ak.array(np.random.randn(100)),
    ...     "d": ak.array(np.random.randn(100))
    ... })
    >>> fig, axes = hist_all(ak_df)

    Save the figure to disk:
    >>> fig, axes = hist_all(ak_df, cols=["a", "b"])
    >>> fig.savefig("hist_all.png")

    """
    if not cols or len(cols) == 0:
        cols = ak_df.columns

    num_rows = int(math.ceil(len(cols) ** 0.5))
    num_cols = (len(cols) + num_rows - 1) // num_rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    fig.tight_layout(pad=2.0)

    if isinstance(axes, plt.Axes):
        axes = np.array([axes])
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, col in enumerate(cols):
        ax = axes[idx]
        try:
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

    return fig, axes
