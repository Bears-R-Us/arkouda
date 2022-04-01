*********************
ArrayView in Arkouda
*********************

.. autoclass:: arkouda.ArrayView

Creation
========
ArrayViews can be created using ak.array or pdarray.reshape

.. code-block:: python

    >>> ak.array([[0, 0], [0, 1], [1, 1]])
    array([[0, 0],
           [0, 1],
           [1, 1]])

    >>> ak.arange(30).reshape(5, 2, 3)
    array([[[ 0,  1,  2],
            [ 3,  4,  5]],

           [[ 6,  7,  8],
            [ 9, 10, 11]],

           [[12, 13, 14],
            [15, 16, 17]],

           [[18, 19, 20],
            [21, 22, 23]],

           [[24, 25, 26],
            [27, 28, 29]]])

Indexing
========
Arkouda ``ArrayView`` objects support basic indexing

* Indexing with an integer ``pdarray`` (of size ndim) or
* Indexing with a mix of integers and slices

Mixed indexing by arrays or "advanced indexing" as numpy calls it is not yet supported. numpy behavior with 2+ arrays is different than equivalent slices.

This example shows how indexing by arrays can be a bit different. This is talked about a bit in https://numpy.org/doc/stable/user/basics.indexing.html

.. code-block:: python

   >>> n = np.arange(4).reshape(2,2)
   # sometimes they line up
   >>> n[:,:]
   array([[0, 1],
          [2, 3]])

   >>> n[:,[0,1]]
   array([[0, 1],
          [2, 3]])

   >>> n[[0,1],:]
   array([[0, 1],
          [2, 3]])
   # sometimes they do not
   >>> n[[0,1],[0,1]]
   array([0, 3])

With 2+ arrays the functionality switches from the Cartesian product of coordinates to more coordinate-wise.

so ``n[:, :]`` gets indices ``[0,0], [0,1], [1,0], [1,1]`` whereas ``n[[0,1],[0,1]]`` only gets indices ``[0,0], [1,1]``

Iteration
===========
Iterating directly over an ``ArrayView`` with ``for x in array_view`` is not supported to discourage transferring all array data from the arkouda server to the Python client. To force this transfer, use the ``to_ndarray`` function to return the ``ArrayView`` as a ``numpy.ndarray``. This transfer will raise an error if it exceeds the byte limit defined in ``arkouda.maxTransferBytes``.

.. autofunction:: arkouda.ArrayView.to_ndarray
