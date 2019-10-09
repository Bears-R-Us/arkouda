**********************
The ``pdarray`` class
**********************

Just as the backbone of NumPy is the ``ndarray``, the backbone of arkouda is an array class called ``pdarray``. And just as the ``ndarray`` object is a Python wrapper for C-style data with C and Fortran methods, the ``pdarray`` object is a Python wrapper for distributed data with parallel methods written in Chapel. The API of ``pdarray`` is similar, but not identical, to that of ``ndarray``.

.. autoclass:: arkouda.pdarray
