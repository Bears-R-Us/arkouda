***********
Indexs in Arkouda
***********

Like Pandas, Arkouda supports ``Indexes``. The purpose and intended functionality remains the same in Arkouda, but are configured to be based on ``arkouda.pdarrays``.

.. autoclass:: arkouda.Index
   :no-index:

Additionally, ``Multi-Indexes`` are supported for indexes with multiple keys.

..autoclass:: arkouda.MultiIndex

Features
==========
``Index`` support the majority of functionality offered by ``pandas.Index``.

Change Dtype
----------
.. autofunction:: arkouda.Index.set_dtype
   :no-index:
.. autofunction:: arkouda.MultiIndex.set_dtype
   :no-index:

ArgSort
----------
.. autofunction:: arkouda.Index.argsort
   :no-index:
.. autofunction:: arkouda.MultiIndex.argsort
   :no-index:

Lookup
----------
.. autofunction:: arkouda.Index.lookup
   :no-index:
.. autofunction:: arkouda.MultiIndex.lookup
   :no-index:

Concat
----------
.. autofunction:: arkouda.Index.concat
   :no-index:
.. autofunction:: arkouda.MultiIndex.concat
   :no-index:
