***********************
Summarizing Data
***********************

Descriptive Statistics
======================

Simple descriptive statistics are available as reduction methods on ``pdarray`` objects. 

.. code-block:: python
		
   >>> A = ak.randint(-10, 11, 1000)
   >>> A.min()
   -10
   >>> A.max()
   10
   >>> A.sum()
   13
   >>> A.mean()
   0.013
   >>> A.var() 
   36.934176000000015
   >>> A.std()
   6.07734942223993

The list of reductions supported on ``pdarray`` objects is:

.. automethod:: arkouda.pdarray.any

.. automethod:: arkouda.pdarray.all

.. automethod:: arkouda.pdarray.is_sorted

.. automethod:: arkouda.pdarray.sum

.. automethod:: arkouda.pdarray.prod

.. automethod:: arkouda.pdarray.min

.. automethod:: arkouda.pdarray.max

.. automethod:: arkouda.pdarray.argmin

.. automethod:: arkouda.pdarray.argmax

.. automethod:: arkouda.pdarray.mean

.. automethod:: arkouda.pdarray.var

.. automethod:: arkouda.pdarray.std

   
Histogram
=========

Arkouda can compute simple histograms on ``pdarray`` data. Currently, this function can only create histograms over evenly spaced bins between the min and max of the data. In the future, we plan to support using a ``pdarray`` to define custom bin edges.

.. autofunction:: arkouda.histogram

Since the ``histogram`` function currently does not return the bin edges, only the counts, the user can recreate the bin edges (e.g. for plotting) using:

.. code-block:: python

   >>> binEdges = np.linspace(myarray.min(), myarray.max(), nbins + 1)


Value Counts
============

For int64 ``pdarray`` objects, it is often useful to count only the unique values that appear. This function finds all unique values and their counts.

.. autofunction:: arkouda.value_counts
