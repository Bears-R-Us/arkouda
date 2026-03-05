.. _setops-label:

********************
Array Set Operations
********************

Following ``numpy.lib.arraysetops``, arkouda supports parallel, distributed set operations using ``pdarray`` objects.

The ``unique`` function effectively converts a ``pdarray`` to a set:

.. autofunction:: arkouda.unique
   :no-index:

.. autofunction:: arkouda.in1d
   :no-index:

.. autofunction:: arkouda.union1d
   :no-index:

.. autofunction:: arkouda.intersect1d
   :no-index:

.. autofunction:: arkouda.setdiff1d
   :no-index:

.. autofunction:: arkouda.setxor1d
   :no-index:
