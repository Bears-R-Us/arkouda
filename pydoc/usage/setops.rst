.. _setops-label:

********************
Array Set Operations
********************

Following ``numpy.lib.arraysetops``, arkouda supports parallel, distributed set operations using ``pdarray`` objects.

The ``unique`` function effectively converts a ``pdarray`` to a set:

.. autofunction:: arkouda.unique

.. autofunction:: arkouda.in1d

.. autofunction:: arkouda.union1d

.. autofunction:: arkouda.intersect1d

.. autofunction:: arkouda.setdiff1d

.. autofunction:: arkouda.setxor1d
