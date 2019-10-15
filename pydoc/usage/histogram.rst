***********************
Examining Distributions
***********************

Histogram
=========

Arkouda can compute simple histograms on ``pdarray`` data. Currently, this function can only create histograms over evenly spaced bins between the min and max of the data. In the future, we plan to support using a ``pdarray`` to define custom bin edges.

.. autofunction:: arkouda.histogram

Since the ``histogram`` function currently does not return the bin edges, only the counts, the user can recreate the bin edges (e.g. for plotting) using:

.. code-block:: python

   binEdges = np.linspace(A.min(), A.max(), nbins + 1)


Value Counts
============

For int64 ``pdarray`` objects, it is often useful to count only the unique values that appear. This function finds all unique values and their counts.

.. autofunction:: arkouda.value_counts
