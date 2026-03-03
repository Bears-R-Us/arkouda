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
   :no-index:

.. automethod:: arkouda.pdarray.all
   :no-index:

.. automethod:: arkouda.pdarray.is_sorted
   :no-index:

.. automethod:: arkouda.pdarray.sum
   :no-index:

.. automethod:: arkouda.pdarray.prod
   :no-index:

.. automethod:: arkouda.pdarray.min
   :no-index:

.. automethod:: arkouda.pdarray.max
   :no-index:

.. automethod:: arkouda.pdarray.argmin
   :no-index:

.. automethod:: arkouda.pdarray.argmax
   :no-index:

.. automethod:: arkouda.pdarray.mean
   :no-index:

.. automethod:: arkouda.pdarray.var
   :no-index:

.. automethod:: arkouda.pdarray.std
   :no-index:

.. automethod:: arkouda.pdarray.mink
   :no-index:

.. automethod:: arkouda.pdarray.maxk
   :no-index:

.. automethod:: arkouda.pdarray.argmink
   :no-index:

.. automethod:: arkouda.pdarray.argmaxk
   :no-index:

   
Histogram
=========

Arkouda can compute simple histograms on ``pdarray`` data. Currently, this function can only create histograms over evenly spaced bins between the min and max of the data. In the future, we plan to support using a ``pdarray`` to define custom bin edges.

.. autofunction:: arkouda.histogram
   :no-index:

Value Counts
============

For int64 ``pdarray`` objects, it is often useful to count only the unique values that appear. This function finds all unique values and their counts.

.. autofunction:: arkouda.value_counts
   :no-index:
