***********
Series in Arkouda
***********

Like Pandas, Arkouda supports ``Series``. The purpose and intended functionality remains the same in Arkouda, but are configured to be based on ``arkouda.pdarrays``.

.. autoclass:: arkouda.Series
   :no-index:

Features
==========
``Series`` support the majority of functionality offered by ``pandas.Series``.

Lookup
----------
.. autofunction:: arkouda.Series.locate
   :no-index:

Lookup
----------
.. autofunction:: arkouda.Series.locate
   :no-index:

Sorting
----------
.. autofunction:: arkouda.Series.sort_index
   :no-index:
.. autofunction:: arkouda.Series.sort_values
   :no-index:

Head/Tail
----------
.. autofunction:: arkouda.Series.topn
   :no-index:
.. autofunction:: arkouda.Series.head
   :no-index:
.. autofunction:: arkouda.Series.tail
   :no-index:

Value Counts
----------
.. autofunction:: arkouda.Series.value_counts
   :no-index:

Pandas Integration
----------
.. autofunction:: arkouda.Series.to_pandas
   :no-index:
.. autofunction:: arkouda.Series.pdconcat
   :no-index:
