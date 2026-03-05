***********
DataFrames in Arkouda
***********

Like Pandas, Arkouda supports ``DataFrames``. The purpose and intended functionality remains the same in Arkouda, but are configured to be based on ``arkouda.pdarrays``.

.. autoclass:: arkouda.DataFrame
   :no-index:

Data Types
==========
Currently, ``DataFrames`` support 4 Arkouda data types for supplying columns.

* ``Arkouda.pdarray``
* ``arkouda.numpy.Strings``
* ``Arkouda.Categorical``
* ``arkouda.numpy.SegArray``

Data within the above objects can be of the types below. Please Note - Not all listed types are compatible with every type above.

* ``int64``: 64-bit signed integer
* ``uint64``: 64-bit unsigned integer
* ``float64``: IEEE 64-bit floating point number
* ``bool``: 8-bit boolean value
* ``str``: Python string

Iteration
=========

Iterating directly over a ``DataFrame`` with ``for x in df`` is not recommended. Doing so is discouraged because it requires transferring all array data from the arkouda server to the Python client since there is almost always a more array-oriented way to express an iterator-based computation. To force this transfer, use the ``to_pandas`` function to return the ``DataFrame`` as a ``pandas.DataFrame``. This transfer will raise an error if it exceeds the byte limit defined in ``ak.client.maxTransferBytes``.

.. autofunction:: arkouda.DataFrame.to_pandas
   :no-index:

Features
==========
``DataFrames`` support the majority of functionality offered by ``pandas.DataFrame``.

Drop
---------
.. autofunction:: arkouda.DataFrame.drop
   :no-index:

GroupBy
----------
.. autofunction:: arkouda.DataFrame.groupby
   :no-index:

Copy
----------
.. autofunction:: arkouda.DataFrame.copy
   :no-index:

Filter
----------
.. autofunction:: arkouda.DataFrame.filter_by_ranges
   :no-index:

Permutations
-------------
.. autofunction:: arkouda.DataFrame.apply_permutation
   :no-index:

Sorting
----------
.. autofunction:: arkouda.DataFrame.argsort
   :no-index:

.. autofunction:: arkouda.DataFrame.coargsort
   :no-index:

.. autofunction:: arkouda.DataFrame.sort_values
   :no-index:

Tail/Head of Data
------------------
.. autofunction:: arkouda.DataFrame.tail
   :no-index:

.. autofunction:: arkouda.DataFrame.head
   :no-index:

Rename Columns
---------------
.. autofunction:: arkouda.DataFrame.rename
   :no-index:

Append
----------
.. autofunction:: akrouda.DataFrame.append
   :no-index:

Concatenate
------------
.. autofunction:: arkouda.DataFrame.concat
   :no-index:

Reset Indexes
--------------
.. autofunction:: arkouda.DataFrame.reset_index
   :no-index:

Deduplication
--------------
.. autofunction:: arkouda.DataFrame.drop_duplicates
   :no-index:
