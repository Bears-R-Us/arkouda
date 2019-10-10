************
Sorting
************

Note: The sorting algorithm in arkouda is currently optimized for a Cray interconnect with a high message rate. For now, sorting runs slowly on Infiniband because of the lower message rate, but upcoming changes to the Chapel runtime involving message buffering should greatly improve sorting speed.

.. autofunction:: arkouda.argsort

.. autofunction:: arkouda.coargsort
