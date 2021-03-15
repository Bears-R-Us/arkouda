.. _groupby-label:

*************
GroupBy
*************

The groupby-aggregate pattern is the workhorse operation in many data science applications, such as feature extraction and graph construction. It relies on ``argsort()`` to group an array of keys and then perform aggregations on other arrays of values.

For example, imagine a dataset with two columns, ``userID`` and ``dayOfWeek``. The following groupby-aggregate operation would show how many user IDs were active on each day of the week:

.. code-block:: python

   # Note: The GroupBy arg should be the values of the dayOfWeek column
   #       and must be an Arkouda compatible data structure i.e. `pdarray`
   byDayOfWeek = ak.GroupBy(data['dayOfWeek'])
   day, numIDs = byDayOfWeek.aggregate(userID, 'nunique')


.. autoclass:: arkouda.GroupBy
   :members:
