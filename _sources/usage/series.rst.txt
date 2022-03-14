***********
Series in Arkouda
***********

Like Pandas, Arkouda supports ``Series``. The purpose and intended functionality remains the same in Arkouda, but are configured to be based on ``arkouda.pdarrays``.

.. autoclass:: arkouda.Series

Features
==========
``Series`` support the majority of functionality offered by ``pandas.Series``.

Lookup
----------
.. autofunction:: arkouda.Series.locate

Lookup
----------
.. autofunction:: arkouda.Series.locate

Sorting
----------
.. autofunction:: arkouda.Series.sort_index
.. autofunction:: arkouda.Series.sort_values

Head/Tail
----------
.. autofunction:: arkouda.Series.topn
.. autofunction:: arkouda.Series.head
.. autofunction:: arkouda.Series.tail

Value Counts
----------
.. autofunction:: arkouda.Series.value_counts

Pandas Integration
----------
.. autofunction:: arkouda.Series.to_pandas
.. autofunction:: arkouda.Series.pdconcat