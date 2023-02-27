**********************
The ``pdarray`` class
**********************

Just as the backbone of NumPy is the ``ndarray``, the backbone of arkouda is an array class called ``pdarray``. And just as the ``ndarray`` object is a Python wrapper for C-style data with C and Fortran methods, the ``pdarray`` object is a Python wrapper for distributed data with parallel methods written in Chapel. The API of ``pdarray`` is similar, but not identical, to that of ``ndarray``.

.. autoclass:: arkouda.pdarray

Data Type
============

Currently, ``pdarray`` supports three user-facing data types (strings are exposed via a separate class, see :ref:`Strings in Arkouda <../strings.rst>`):

* ``int64``: 64-bit signed integer
* ``float64``: IEEE 64-bit floating point number
* ``bool``: 8-bit boolean value

Arkouda inherits all of its data types from numpy. For example, ``ak.int64`` is derived from ``np.int64``.

Rank
=============

Currently, a ``pdarray`` can only have rank 1. We plan to support sparse, multi-dimensional arrays via data structures incorporating rank-1 ``pdarray`` objects.

Name
============

The ``name`` attribute of an array is a string used by the arkouda server to identify the ``pdarray`` object in its symbol table. This name is chosen by the server, and the user should not overwrite it.

Operators
=========

The ``pdarray`` class supports most Python special methods, including arithmetic, bitwise, and comparison operators.

Iteration
=========

Iterating directly over a ``pdarray`` with ``for x in array`` is not supported to discourage transferring all array data from the arkouda server to the Python client since there is almost always a more array-oriented way to express an iterator-based computation. To force this transfer, use the ``to_ndarray`` function to return the ``pdarray`` as a ``numpy.ndarray``. This transfer will raise an error if it exceeds the byte limit defined in ``ak.client.maxTransferBytes``.

.. autofunction:: arkouda.pdarray.to_ndarray


.. _cast-label:

Type Casting
============

Conversion between dtypes is sometimes implicit, as in the following example:

.. code-block:: python

   >>> a = ak.arange(10)
   >>> b = 1.0 * a
   >>> b.dtype
   dtype('float64')

Explicit conversion is supported via the ``cast`` function.

.. autofunction:: arkouda.cast

Reshape
=======

Using the ``.reshape`` method, a multi-dimension view of a pdarray will be returned as an ``ArrayView``

.. autofunction:: arkodua.pdarray.reshape
