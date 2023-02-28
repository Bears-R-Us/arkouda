*********************
Categoricals
*********************

Categorical arrays are a concept from Pandas that speeds up many operations on strings, especially when an array of strings contains many repeated values. A ``Categorical`` object stores the unique strings as category labels and represents the values of the original array as integer indices into this category array.

Construction
============

The typical way to construct a ``Categorical`` is from a ``Strings`` object:

  .. autoclass:: arkouda.Categorical

However, if one already has pre-computed unique categories and integer indices, the following constructor is useful:

  .. automethod:: arkouda.Categorical.from_codes

Operations
==========
                  
Arkouda ``Categorical`` objects support all operations that ``Strings`` support, and they will almost always execute faster:

* Indexing with integer, slice, integer ``pdarray``, and boolean ``pdarray`` (see :ref:`indexing-label`)
* Comparison (``==`` and ``!=``) with string literal or other ``Categorical`` object of same size
* Substring search
  
  .. automethod:: arkouda.Categorical.contains
                    
  .. automethod:: arkouda.Categorical.startswith
                    
  .. automethod:: arkouda.Categorical.endswith
                    
* :ref:`setops-label`, e.g. ``unique`` and ``in1d``
* :ref:`sorting-label`, via ``argsort`` and ``coargsort``
* :ref:`groupby-label`, both alone and in conjunction with numeric arrays

Iteration
=========

Iterating directly over a ``Categorical`` with ``for x in categorical`` is not supported to discourage transferring all the ``Categorical`` object's data from the arkouda server to the Python client since there is almost always a more array-oriented way to express an iterator-based computation. To force this transfer, use the ``to_ndarray`` function to return the ``categorical`` as a ``numpy.ndarray``. This transfer will raise an error if it exceeds the byte limit defined in ``ak.client.maxTransferBytes``.

.. autofunction:: arkouda.Categorical.to_ndarray
