************************
Indexing and Assignment
************************

Arkouda ``pdarray`` objects support the same indexing and assignment syntax as rank-1 NumPy arrays.

Integer
=======

Indexing and assigment with a single integer work the same as in Python.

.. code-block:: python

   >>> A = ak.arange(0, 10, 1)
   >>> A[5]
   5
   >>> A[5] = 42
   >>> A[5]
   42

Slice
=======

Indexing and assignment are also supported via Python-like slices. A Python slice has a start (inclusive), stop (exclusive), and stride. All three of these parameters can be implied; the default start is the beginning of the array (0 for positive strides, -1 for negative), the default stop is the end of the array (``len`` for positive strides, -1 for negative), and the default stride is 1.

.. code-block:: python

   >>> A = ak.arange(0, 10, 1)
   >>> A[2:6]
   array([2, 3, 4, 5])
   >>> A[::2]
   array([0, 2, 4, 6, 8])
   >>> A[3::-1]
   array([3, 2, 1, 0])
   >>> A[1::2] = ak.zeros(5)
   >>> A
   array([0, 0, 2, 0, 4, 0, 6, 0, 8, 0])

Gather/Scatter (``pdarray``)
============================

Gather and scatter operations can be expressed using a ``pdarray`` as an index to another ``pdarray``.

Integer ``pdarray`` index
-------------------------

With an integer ``pdarray``, you can gather a list of indices from the target array. The indices can be out of order and non-unique. For assignment, the right-hand side must be a ``pdarray`` the same size as the index array.

.. code-block:: python

   >>> A = ak.arange(10, 20, 1)
   >>> inds = ak.array([8, 2, 5])
   >>> A[inds]
   array([18, 12, 15])
   >>> A[inds] = ak.zeros(3)
   >>> A
   array([10, 11, 0, 13, 14, 0, 16, 17, 0, 19])

Logical indexing
----------------

Logical indexing is a powerful construct from NumPy (and Matlab). In logical indexing, the index must be a ``pdarray`` of type ``bool`` that is the same size as the outer ``pdarray`` being indexed. The indexing only touches those elements of the outer ``pdarray`` where the corresponding element of the index ``pdarray`` is ``True``. 

.. code-block:: python

   >>> A = ak.arange(0, 10, 1)
   >>> inds = ak.zeros(10, dtype=ak.bool)
   >>> inds[2] = True
   >>> inds[5] = True
   >>> A[inds]
   array([2, 5])
   ..
      >>> A[inds] = 42
      >>> A
      array([0, 1, 42, 3, 4, 42, 6, 7, 8, 9])

Assignment with a logical index vector is not supported, but will be soon.
