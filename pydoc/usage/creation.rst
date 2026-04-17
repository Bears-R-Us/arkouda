***************
Creating Arrays
***************

There are several ways to initialize arkouda ``pdarray`` objects, most of which come from NumPy.

Constant
========

.. autofunction:: arkouda.zeros
   :no-index:

.. autofunction:: arkouda.ones
   :no-index:

.. autofunction:: arkouda.zeros_like
   :no-index:

.. autofunction:: arkouda.ones_like
   :no-index:


Regular
=======

.. autofunction:: arkouda.arange
   :no-index:

.. autofunction:: arkouda.linspace
   :no-index:

Random
======

.. autofunction:: arkouda.randint
   :no-index:

.. _concatenate-label:
   :no-index:
                  
Concatenation
=============

Performance note: in multi-locale settings, the default (ordered) mode of ``concatenate`` is very communication-intensive because the distribution of the original and resulting arrays are unrelated and most data must be moved non-locally. If the application does not require the concatenated array to be ordered (e.g. if the result is simply going to be sorted anyway), then using the keyword ``ordered=False`` will greatly speed up concatenation by minimizing non-local data movement.

.. autofunction:: arkouda.concatenate
   :no-index:
