*********************************
Arithmetic and Numeric Operations
*********************************

Vector and Scalar Arithmetic
============================

A large subset of Python's binary and in-place operators are supported on ``pdarray`` objects. Where supported, the behavior of these operators is identical to that of NumPy ``ndarray`` objects.

.. code-block:: python

   >>> A = ak.arange(10)
   >>> A += 2
   >>> A
   array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
   >>> A + A
   array([4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
   >>> 2 * A
   array([4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
   >>> A == A
   array([True, True, True, True, True, True, True, True, True, True])


Operations that are not implemented will raise a ``RuntimeError``. In-place operations that would change the dtype of the ``pdarray`` are not implemented.

Element-wise Functions
======================

Arrays support several mathematical functions that operate element-wise and return a ``pdarray`` of the same length.

.. autofunction:: arkouda.abs

.. autofunction:: arkouda.log

.. autofunction:: arkouda.exp

.. autofunction:: arkouda.sin

.. autofunction:: arkouda.cos

Scans
========

Scans perform a cumulative reduction over a ``pdarray``, returning a ``pdarray`` of the same size.

.. autofunction:: arkouda.cumsum

.. autofunction:: arkouda.cumprod

Reductions
==========

Reductions return a scalar value.
		  
.. autofunction:: arkouda.any

.. autofunction:: arkouda.all

.. autofunction:: arkouda.is_sorted

.. autofunction:: arkouda.sum

.. autofunction:: arkouda.prod

.. autofunction:: arkouda.min

.. autofunction:: arkouda.max

.. autofunction:: arkouda.argmin

.. autofunction:: arkouda.argmax

.. autofunction:: arkouda.mean

.. autofunction:: arkouda.var

.. autofunction:: arkouda.std

Where
=====

The ``where`` function is a way to multiplex two ``pdarray`` (or a ``pdarray`` and a scalar) based on a condition:

.. autofunction:: arkouda.where
