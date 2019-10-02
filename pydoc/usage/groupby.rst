*************
GroupBy
*************

The groupby-aggregate pattern is the workhorse operation in many data science applications, such as feature extraction and graph construction. It relies on ``argsort()`` to group an array of keys and then perform aggregations on other arrays of values.

For example, imagine a dataset with two columns, ``userID`` and ``dayOfWeek``. The following groupby-aggregate operation would show how many user IDs were active on each day of the week:

.. code-block:: python

   byDayOfWeek = ak.GroupBy(dayOfWeek)
   day, numIDs = byDayOfWeek.aggregate(userID, 'nunique')


.. autoclass:: arkouda.GroupBy
   :members: count, aggregate
