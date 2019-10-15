**********************
The ``pdarray`` class
**********************

Just as the backbone of NumPy is the ``ndarray``, the backbone of arkouda is an array class called ``pdarray``. And just as the ``ndarray`` object is a Python wrapper for C-style data with C and Fortran methods, the ``pdarray`` object is a Python wrapper for distributed data with parallel methods written in Chapel. The API of ``pdarray`` is similar, but not identical, to that of ``ndarray``.

.. autoclass:: arkouda.pdarray

Data Type
============

Currently, ``pdarray`` supports three data types:

* ``int64``: 64-bit signed integer
* ``float64``: IEEE 64-bit floating point number
* ``bool``: 8-bit boolean value

Arkouda inherits all of its data types from numpy. For example, ``ak.int64`` is assigned to ``np.int64``.

Rank
=============

Currently, a ``pdarray`` can only have rank 1. We plan to support sparse, multi-dimensional arrays via data structures incorporating rank-1 ``pdarray`` objects.

Name
============

The ``name`` attribute of an array is a string used by the arkouda server to identify the ``pdarray`` object in its symbol table. This name is chosen by the server, and the user should not overwrite it.
