***********
Indexs in Arkouda
***********

Like Pandas, Arkouda supports ``Indexes``. The purpose and intended functionality remains the same in Arkouda, but are configured to be based on ``arkouda.pdarrays``.

.. autoclass:: arkouda.Index

Additionally, ``Multi-Indexes`` are supported for indexes with multiple keys.

..autoclass:: arkouda.MultiIndex

Features
==========
``Index`` support the majority of functionality offered by ``pandas.Index``.

Change Dtype
----------
.. autofunction:: arkouda.Index.set_dtype
.. autofunction:: arkouda.MultiIndex.set_dtype

ArgSort
----------
.. autofunction:: arkouda.Index.argsort
.. autofunction:: arkouda.MultiIndex.argsort

Lookup
----------
.. autofunction:: arkouda.Index.lookup
.. autofunction:: arkouda.MultiIndex.lookup

Concat
----------
.. autofunction:: arkouda.Index.concat
.. autofunction:: arkouda.MultiIndex.concat
