***********
DataFrames in Arkouda
***********

Like Pandas, Arkouda supports ``DataFrames``. The purpose and intended functionality remains the same in Arkouda, but are configured to be based on ``arkouda.pdarrays``.

.. autoclass:: arkouda.DataFrame

Data Types
==========
Currently, ``DataFrames`` support 4 Arkouda data types for supplying columns.

* ``Arkouda.pdarray``
* ``Arkouda.Strings``
* ``Arkouda.Categorical``
* ``Arkouda.SegArray``

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

Features
==========
``DataFrames`` support the majority of functionality offered by ``pandas.DataFrame``.

Drop
---------
.. autofunction:: arkouda.DataFrame.drop

GroupBy
----------
.. autofunction:: arkouda.DataFrame.groupby

Copy
----------
.. autofunction:: arkouda.DataFrame.copy

Filter
----------
.. autofunction:: arkouda.DataFrame.filter_by_ranges

Permutations
-------------
.. autofunction:: arkouda.DataFrame.apply_permutation

Sorting
----------
.. autofunction:: arkouda.DataFrame.argsort

.. autofunction:: arkouda.DataFrame.coargsort

.. autofunction:: arkouda.DataFrame.sort_values

Tail/Head of Data
------------------
.. autofunction:: arkouda.DataFrame.tail

.. autofunction:: arkouda.DataFrame.head

Rename Columns
---------------
.. autofunction:: arkouda.DataFrame.rename

Append
----------
.. autofunction:: akrouda.DataFrame.append

Concatenate
------------
.. autofunction:: arkouda.DataFrame.concat

Reset Indexes
--------------
.. autofunction:: arkouda.DataFrame.reset_index

Deduplication
--------------
.. autofunction:: arkouda.DataFrame.drop_duplicates
