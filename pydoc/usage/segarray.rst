*********************
SegArrays in Arkouda
*********************

In NumPy, arrays containing variable-length sub-arrays are supported as an array containing a single column. Each column contains another ``ndarray`` of some length. Depending on the chosen approach in NumPy, this can result in a loss of functioanlity. In Arkouda, the array containing variable-length sub-arrays is its own class: ``SegArray``

In order to efficiently store arrays with varying row and column dimensions, Arkouda uses a "segmented array" data strucuture:

* ``segments``: An ``int64`` array containing the start index of each sub-array within the flattened values array
* ``values``: The flattened values of all sub-arrays


Performance
===========
``SegArray`` objects are currently processed entire on the Arkouda client side. The data structure is reflective of the data structure that will be used for Arkouda server side processing.

Iteration
===========
Because ``SegArray`` is currently processing entirely on the Arkouda client side, iteration is natively supported. Thus, ``for row in segarr`` with iterate over each sub-array. Each of these sub-arrays is currently returned as a ``numpy.ndarray``.

Similar to ``Strings``, ``SegArrays`` will be moved to process server side. This will remove the ability to natively iterate to discourage transferring all of the objects data to the client. In order to support this moving forward, ``SegArray`` includes a ``to_ndarray()`` function. It is recommended that this function be used for iteration over ``SegArray`` objects, to prevent issues associated with moving processing server side. For more information on the usage of ``to_ndarray`` with SegArray

.. autofunction:: arkouda.numpy.SegArray.to_ndarray

Operation
===========
Arkouda ``SegArray`` objects support the following operations:

* Indexing with integer, slice, integer ``pdarray``, and boolean ``pdarray`` (see :ref:`indexing-label`)
* Comparison (==) Provides an Arkouda ``pdarray`` containing ``bool`` values indicating the equality of each sub-array in the ``SegArray``.
* :ref:`setops-label`, e.g. ``unique``
* :ref:`concatenate-label` with other ``SegArrays``. Horizontal and vertical axis supported.

SegArray Specific Methods
===========

Prefix & Suffix
-----------
.. autofunction:: arkouda.numpy.SegArray.get_prefixes

.. autofunction:: arkouda.numpy.SegArray.get_suffixes

NGrams
----------
.. autofunction:: arkouda.numpy.SegArray.get_ngrams

Sub-array of Size
----------
.. autofunction:: arkouda.numpy.SegArray.get_length_n

Access/Set Specific Elements in Sub-Array
----------
.. autofunction:: arkouda.numpy.SegArray.get_jth

.. autofunction:: arkouda.numpy.SegArray.set_jth

Append & Prepend
----------
.. autofunction:: arkouda.numpy.SegArray.append

.. autofunction:: arkouda.numpy.SegArray.append_single

.. autofunction:: arkouda.numpy.SegArray.prepend_single

Deduplication
----------
.. autofunction:: arkouda.numpy.SegArray.remove_repeats

SegArray SetOps
===============

Union
-----
.. autofunction:: arkouda.numpy.SegArray.union

Intersect
---------
.. autofunction:: arkouda.numpy.SegArray.intersect

Set Difference
--------------
.. autofunction:: arkouda.numpy.SegArray.setdiff

Symmetric Difference
--------------------
.. autofunction:: arkouda.numpy.SegArray.setxor
