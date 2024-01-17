from __future__ import annotations

import json
import os
import random
from collections import UserDict
from typing import Callable, Dict, List, Optional, Union, cast
from warnings import warn

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typeguard import typechecked

from arkouda.categorical import Categorical
from arkouda.client import generic_msg, maxTransferBytes
from arkouda.client_dtypes import BitVector, Fields, IPv4
from arkouda.dtypes import BigInt
from arkouda.dtypes import bool as akbool
from arkouda.dtypes import float64 as akfloat64
from arkouda.dtypes import int64 as akint64
from arkouda.dtypes import uint64 as akuint64
from arkouda.groupbyclass import GROUPBY_REDUCTION_TYPES
from arkouda.groupbyclass import GroupBy as akGroupBy
from arkouda.groupbyclass import unique
from arkouda.index import Index, MultiIndex
from arkouda.join import inner_join
from arkouda.numeric import cast as akcast
from arkouda.numeric import cumsum, where
from arkouda.pdarrayclass import RegistrationError, pdarray
from arkouda.pdarraycreation import arange, array, create_pdarray, full, zeros
from arkouda.pdarraysetops import concatenate, in1d, intersect1d
from arkouda.row import Row
from arkouda.segarray import SegArray
from arkouda.series import Series
from arkouda.sorting import argsort, coargsort
from arkouda.strings import Strings
from arkouda.timeclass import Datetime, Timedelta

# This is necessary for displaying DataFrames with BitVector columns,
# because pandas _html_repr automatically truncates the number of displayed bits
pd.set_option("display.max_colwidth", 65)

__all__ = [
    "DataFrame",
    "sorted",
    "intersect",
    "invert_permutation",
    "intx",
    "merge",
]


def groupby_operators(cls):
    for name in GROUPBY_REDUCTION_TYPES:
        setattr(cls, name, cls._make_aggop(name))
    return cls


@groupby_operators
class GroupBy:
    """
    A DataFrame that has been grouped by a subset of columns

    Parameters
    ----------

    gb_key_names : str or list(str), default=None
        The column name(s) associated with the aggregated columns.

    as_index : bool, default=True
        If True, interpret aggregated column as index
        (only implemented for single dimensional aggregates).
        Otherwise, treat aggregated column as a dataframe column.


    Attributes
    ----------
    gb :    arkouda.groupbyclass.GroupBy
        GroupBy object, where the aggregation keys are values of column(s) of a dataframe,
        usually in preparation for aggregating with respect to the other columns.
    df :    arkouda.dataframe.DataFrame
        The dataframe containing the original data.
    gb_key_names    :    str or list(str)
        The column name(s) associated with the aggregated columns.
    as_index : bool (default=True)
        If True the grouped values of the aggregation keys will be treated as an index.
    dropna : bool (default=True)
        If True, and the groupby keys contain NaN values,
        the NaN values together with the corresponding row will be dropped.
        Otherwise, the rows corresponding to NaN values will be kept.
    """

    def __init__(self, gb, df, gb_key_names=None, as_index=True):
        self.gb = gb
        self.df = df
        self.gb_key_names = gb_key_names
        self.as_index = as_index
        for attr in ["nkeys", "permutation", "unique_keys", "segments"]:
            setattr(self, attr, getattr(gb, attr))

    @classmethod
    def _make_aggop(cls, opname):
        numerical_dtypes = [akfloat64, akint64, akuint64]

        def aggop(self, colnames=None):
            """
            Aggregate the operation, with the grouped column(s) values as keys.

            Parameters
            ----------

            colnames : (list of) str, default=None
                Column name or list of column names to compute the aggregation over.

            Returns
            -------
            arkouda.dataframe.DataFrame

            """
            if colnames is None:
                colnames = list(self.df.data.keys())
            elif isinstance(colnames, str):
                colnames = [colnames]
            colnames = [
                c
                for c in colnames
                if (
                    (self.df.data[c].dtype.type in numerical_dtypes)
                    or isinstance(self.df.data[c].dtype, BigInt)
                )
                and (
                    (isinstance(self.gb_key_names, str) and (c != self.gb_key_names))
                    or (isinstance(self.gb_key_names, list) and c not in self.gb_key_names)
                )
            ]

            if isinstance(colnames, List):
                if isinstance(self.gb_key_names, str):
                    return DataFrame(
                        {c: self.gb.aggregate(self.df.data[c], opname)[1] for c in colnames},
                        index=Index(self.gb.unique_keys, name=self.gb_key_names),
                    )
                elif isinstance(self.gb_key_names, list) and len(self.gb_key_names) == 1:
                    return DataFrame(
                        {c: self.gb.aggregate(self.df.data[c], opname)[1] for c in colnames},
                        index=Index(self.gb.unique_keys, name=self.gb_key_names[0]),
                    )
                elif isinstance(self.gb_key_names, list):
                    column_dict = dict(zip(self.gb_key_names, self.unique_keys))
                    for c in colnames:
                        column_dict[c] = self.gb.aggregate(self.df.data[c], opname)[1]
                    return DataFrame(column_dict)
                else:
                    return None

        return aggop

    def count(self, as_series=None):
        """
        Compute the count of each value as the total number of rows, including NaN values.
        This is an alias for size(), and may change in the future.

        Parameters
        ----------

        as_series : bool, default=None
            Indicates whether to return arkouda.dataframe.DataFrame (if as_series = False) or
            arkouda.series.Series (if as_series = True)

        Returns
        -------

        arkouda.dataframe.DataFrame (if as_series = False) or
        arkouda.series.Series (if as_series = True)

        """
        if as_series is True or (as_series is None and self.as_index is True):
            return self.__return_agg_series(self.gb.count())
        else:
            return self.__return_agg_dataframe(self.gb.count(), "count")

    def size(self, as_series=None, sort_index=True):
        """
        Compute the size of each value as the total number of rows, including NaN values.

        Parameters
        ----------

        as_series : bool, default=None
            Indicates whether to return arkouda.dataframe.DataFrame (if as_series = False) or
            arkouda.series.Series (if as_series = True)
        sort_index : bool, default=True
            If True, results will be returned with index values sorted in ascending order.

        Returns
        -------

        arkouda.dataframe.DataFrame or
        arkouda.series.Series

        """
        if as_series is True or (as_series is None and self.as_index is True):
            return self.__return_agg_series(self.gb.size(), sort_index=sort_index)
        else:
            return self.__return_agg_dataframe(self.gb.size(), "size", sort_index=sort_index)

    def __return_agg_series(self, values, sort_index=True):
        if self.as_index is True:
            if isinstance(self.gb_key_names, str):
                series = Series(values, index=Index(self.gb.unique_keys, name=self.gb_key_names))
            elif isinstance(self.gb_key_names, list) and len(self.gb_key_names) == 1:
                series = Series(values, index=Index(self.gb.unique_keys, name=self.gb_key_names[0]))
            elif isinstance(self.gb_key_names, list) and len(self.gb_key_names) > 1:
                from arkouda.index import MultiIndex

                series = Series(
                    values,
                    index=MultiIndex(self.gb.unique_keys, names=self.gb_key_names),
                )
        else:
            series = Series(values)

        if sort_index is True:
            series = series.sort_index()

        return series

    def __return_agg_dataframe(self, values, name, sort_index=True):
        if isinstance(self.gb_key_names, str):
            if self.as_index is True:
                df = DataFrame(
                    {name: values[1]},
                    index=Index(self.gb.unique_keys, name=self.gb_key_names),
                )
            else:
                df = DataFrame({self.gb_key_names: self.gb.unique_keys, name: values[1]})

            if sort_index is True:
                df = df.sort_index()

            return df

        elif len(self.gb_key_names) == 1:
            if self.as_index is True:
                df = DataFrame(
                    {name: values[1]},
                    index=Index(self.gb.unique_keys, name=self.gb_key_names[0]),
                )
            else:
                df = DataFrame(
                    {self.gb_key_names[0]: self.gb.unique_keys, name: values[1]},
                )

            if sort_index is True:
                df = df.sort_index()

            return df
        else:
            return Series(values).to_dataframe(index_labels=self.gb_key_names, value_label=name)

    def diff(self, colname):
        """Create a difference aggregate for the given column

        For each group, the difference between successive values is calculated.
        Aggregate operations (mean,min,max,std,var) can be done on the results.

        Parameters
        ----------

        colname:  String. Name of the column to compute the difference on.

        Returns
        -------

        DiffAggregate : object containing the differences, which can be aggregated.

        """

        return DiffAggregate(self.gb, self.df.data[colname])

    def broadcast(self, x, permute=True):
        """Fill each group’s segment with a constant value.

        Parameters
        ----------

        x :  Either a Series or a pdarray

        Returns
        -------
        A aku.Series with the Index of the original frame and the values of the broadcast.
        """

        if isinstance(x, Series):
            data = self.gb.broadcast(x.values, permute=permute)
        else:
            data = self.gb.broadcast(x, permute=permute)
        return Series(data=data, index=self.df.index)


@groupby_operators
class DiffAggregate:
    """
    A column in a GroupBy that has been differenced.
    Aggregation operations can be done on the result.
    """

    def __init__(self, gb, series):
        self.gb = gb

        values = zeros(len(series), "float64")
        series_permuted = series[gb.permutation]
        values[1:] = akcast(series_permuted[1:] - series_permuted[:-1], "float64")
        values[gb.segments] = np.nan
        self.values = values

    @classmethod
    def _make_aggop(cls, opname):
        def aggop(self):
            return Series(self.gb.aggregate(self.values, opname))

        return aggop


"""
DataFrame structure based on Arkouda arrays.
"""


class DataFrame(UserDict):
    """
    A DataFrame structure based on arkouda arrays.

    Parameters
    ----------
    initialdata : List or dictionary of lists, tuples, or pdarrays
        Each list/dictionary entry corresponds to one column of the data and
        should be a homogenous type. Different columns may have different
        types. If using a dictionary, keys should be strings.

    index : Index, pdarray, or Strings
        Index for the resulting frame. Defaults to an integer range.

    columns : List, tuple, pdarray, or Strings
        Column labels to use if the data does not include them. Elements must
        be strings. Defaults to an stringified integer range.

    Examples
    --------

    Create an empty DataFrame and add a column of data:

    >>> import arkouda as ak
    >>> import numpy as np
    >>> import pandas as pd
    >>> df = ak.DataFrame()
    >>> df['a'] = ak.array([1,2,3])

    Create a new DataFrame using a dictionary of data:

    >>> userName = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
    >>> userID = ak.array([111, 222, 111, 333, 222, 111])
    >>> item = ak.array([0, 0, 1, 1, 2, 0])
    >>> day = ak.array([5, 5, 6, 5, 6, 6])
    >>> amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
    >>> df = ak.DataFrame({'userName': userName, 'userID': userID,
    >>>            'item': item, 'day': day, 'amount': amount})
    >>> df
    DataFrame(['userName', 'userID', 'item', 'day', 'amount'] [6 rows : 224 B])

    Indexing works slightly differently than with pandas:
    >>> df[0]
    {'userName': 'Alice', 'userID': 111, 'item': 0, 'day': 5, 'amount': 0.5}
    >>> df['userID']
    array([111, 222, 111, 333, 222, 111])
    >>> df['userName']
    array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
    >>> df[[1,5,7]]
      userName  userID  item  day  amount
    1      Bob     222     0    5     0.6
    2    Alice     111     1    6     1.1
    3    Carol     333     1    5     1.2

    Note that strides are not implemented except for stride = 1.
    >>> df[1:5:1]
    DataFrame(['userName', 'userID', 'item', 'day', 'amount'] [4 rows : 148 B])
    >>> df[ak.array([1,2,3])]
    DataFrame(['userName', 'userID', 'item', 'day', 'amount'] [3 rows : 112 B])
    >>> df[['userID', 'day']]
    DataFrame(['userID', 'day'] [6 rows : 96 B])
    """

    COLUMN_CLASSES = (pdarray, Strings, Categorical, SegArray)

    objType = "DataFrame"

    def __init__(self, initialdata=None, index=None, columns=None):
        super().__init__()
        self.registered_name = None

        if isinstance(initialdata, DataFrame):
            # Copy constructor
            self._size = initialdata._size
            self._bytes = initialdata._bytes
            self._empty = initialdata._empty
            self._columns = initialdata._columns
            if index is None:
                self._set_index(initialdata.index)
            else:
                self._set_index(index)
            self.data = initialdata.data
            self.update_size()
            return
        elif isinstance(initialdata, pd.DataFrame):
            # copy pd.DataFrame data into the ak.DataFrame object
            self._size = initialdata.size
            self._bytes = 0
            self._empty = initialdata.empty
            self._columns = initialdata.columns.tolist()

            if index is None:
                self._set_index(initialdata.index.values.tolist())
            else:
                self._set_index(index)
            self.data = {}
            for key in initialdata.columns:
                self.data[key] = (
                    SegArray.from_multi_array([array(r) for r in initialdata[key]])
                    if isinstance(initialdata[key][0], (list, np.ndarray))
                    else array(initialdata[key])
                )

            self.data.update()
            return

        # Some metadata about this dataframe.
        self._size = 0
        self._bytes = 0
        self._empty = True

        # Initial attempts to keep an order on the columns
        self._columns = []
        self._set_index(index)

        # Add data to the DataFrame if there is any
        if initialdata is not None:
            # Used to prevent uneven array length in initialization.
            sizes = set()

            # Initial data is a dictionary of arkouda arrays
            if isinstance(initialdata, dict):
                for key, val in initialdata.items():
                    if isinstance(val, (list, tuple)):
                        val = array(val)
                    if not isinstance(val, self.COLUMN_CLASSES):
                        raise ValueError(f"Values must be one of {self.COLUMN_CLASSES}.")
                    if key.lower() == "index":
                        # handles the index as an Index object instead of a column
                        self._set_index(val)
                        continue
                    sizes.add(val.size)
                    if len(sizes) > 1:
                        raise ValueError("Input arrays must have equal size.")
                    self._empty = False
                    UserDict.__setitem__(self, key, val)
                    # Update the column index
                    self._columns.append(key)

            # Initial data is a list of arkouda arrays
            elif isinstance(initialdata, list):
                # Create string IDs for the columns
                keys = []
                if columns is not None:
                    if any(not isinstance(label, str) for label in columns):
                        raise TypeError("Column labels must be strings.")
                    if len(columns) != len(initialdata):
                        raise ValueError("Must have as many labels as columns")
                    keys = columns
                else:
                    keys = [str(x) for x in range(len(initialdata))]

                for key, col in zip(keys, initialdata):
                    if isinstance(col, (list, tuple)):
                        col = array(col)
                    if not isinstance(col, self.COLUMN_CLASSES):
                        raise ValueError(f"Values must be one of {self.COLUMN_CLASSES}.")
                    sizes.add(col.size)
                    if len(sizes) > 1:
                        raise ValueError("Input arrays must have equal size.")
                    self._empty = False
                    UserDict.__setitem__(self, key, col)
                    # Update the column index
                    self._columns.append(key)

            # Initial data is invalid.
            else:
                raise ValueError(f"Initialize with dict or list of {self.COLUMN_CLASSES}.")

            # Update the dataframe indices and metadata.
            if len(sizes) > 0:
                self._size = sizes.pop()

            # If the index param was passed in, use that instead of
            # creating a new one.
            if self.index is None:
                self._set_index(arange(self._size))
            else:
                self._set_index(index)
            self.update_size()

    def __getattr__(self, key):
        if key not in self.columns:
            raise AttributeError(f"Attribute {key} not found")
        # Should this be cached?
        return Series(data=self[key], index=self.index.index)

    def __dir__(self):
        return dir(DataFrame) + self.columns

    # delete a column
    def __delitem__(self, key):
        # This function is a backdoor to messing up the indices and columns.
        # I needed to reimplement it to prevent bad behavior
        UserDict.__delitem__(self, key)
        self._columns.remove(key)

        # If removing this column emptied the dataframe
        if len(self._columns) == 0:
            self._set_index(None)
            self._empty = True
        self.update_size()

    def __getitem__(self, key):
        # convert series to underlying values
        # Should check for index alignment
        if isinstance(key, Series):
            key = key.values

        # Select rows using an integer pdarray
        if isinstance(key, pdarray):
            if key.dtype == akbool:
                key = arange(key.size)[key]
            result = {}
            for k in self._columns:
                result[k] = UserDict.__getitem__(self, k)[key]
            # To stay consistent with numpy, provide the old index values
            return DataFrame(initialdata=result, index=self.index.index[key])

        # Select rows or columns using a list
        if isinstance(key, (list, tuple)):
            result = DataFrame()
            if len(key) <= 0:
                return result
            if len({type(x) for x in key}) > 1:
                raise TypeError("Invalid selector: too many types in list.")
            if isinstance(key[0], str):
                for k in key:
                    result.data[k] = UserDict.__getitem__(self, k)
                    result._columns.append(k)
                result._empty = False
                result._set_index(self.index)  # column lens remain the same. Copy the indexing
                return result
            else:
                raise TypeError(
                    "DataFrames only support lists for column indexing. "
                    "All list entries must be of type str."
                )

        # Select a single row using an integer
        if isinstance(key, int):
            result = {}
            row = array([key])
            for k in self._columns:
                result[k] = (UserDict.__getitem__(self, k)[row])[0]
            return Row(result)

        # Select a single column using a string
        elif isinstance(key, str):
            if key not in self.keys():
                raise KeyError(f"Invalid column name '{key}'.")
            return UserDict.__getitem__(self, key)

        # Select rows using a slice
        elif isinstance(key, slice):
            # result = DataFrame()
            rtn_data = {}
            s = key
            for k in self._columns:
                rtn_data[k] = UserDict.__getitem__(self, k)[s]
            return DataFrame(initialdata=rtn_data, index=self.index.index[arange(self.size)[s]])
        else:
            raise IndexError("Invalid selector: unknown error.")

    def __setitem__(self, key, value):
        self.update_size()

        # If this is the first column added, we must create an index column.
        add_index = False
        if self._empty:
            add_index = True

        # Set a single row in the dataframe using a dict of values
        if isinstance(key, int):
            for k in self._columns:
                if isinstance(self.data[k], Strings):
                    raise ValueError(
                        "This DataFrame has a column of type ak.Strings;"
                        " so this DataFrame is immutable. This feature could change"
                        " if arkouda supports mutable Strings in the future."
                    )
            if self._empty:
                raise ValueError("Initial data must be dict of arkouda arrays.")
            elif not isinstance(value, (dict, UserDict)):
                raise ValueError("Expected dict or Row type.")
            elif key >= self._size:
                raise KeyError("The row index is out of range.")
            else:
                for k, v in value.items():
                    # maintaining to prevent adding index column
                    if k == "index":
                        continue
                    self[k][key] = v

        # Set a single column in the dataframe using a an arkouda array
        elif isinstance(key, str):
            if not isinstance(value, self.COLUMN_CLASSES):
                raise ValueError(f"Column must be one of {self.COLUMN_CLASSES}.")
            elif self._size is not None and self._size != value.size:
                raise ValueError(f"Expected size {self.size} but received size {value.size}.")
            else:
                self._empty = False
                UserDict.__setitem__(self, key, value)
                # Update the index values
                if key not in self._columns:
                    self._columns.append(key)

        # Do nothing and return if there's no valid data
        else:
            raise ValueError("No valid data received.")

        # Update the dataframe indices and metadata.
        if add_index:
            self.update_size()
            self._set_index(arange(self._size))

    def __len__(self):
        """
        Return the number of rows
        """
        return self.size

    def _ncols(self):
        """
        Number of columns.
        If index appears, we now want to utilize this
        because the actual index has been moved to a property
        """
        return len(self._columns)

    def __str__(self):
        """
        Returns a summary string of this dataframe.
        """

        self.update_size()

        if self._empty:
            return "DataFrame([ -- ][ 0 rows : 0 B])"

        keys = [str(key) for key in list(self._columns)]
        keys = [("'" + key + "'") for key in keys]
        keystr = ", ".join(keys)

        # first call to memory_usage() initializes self._bytes
        mem = self.memory_usage()

        # Get units that make the most sense.
        if self._bytes < 1024:
            mem = self.memory_usage(unit="B")
        elif self._bytes < 1024**2:
            mem = self.memory_usage(unit="KB")
        elif self._bytes < 1024**3:
            mem = self.memory_usage(unit="MB")
        else:
            mem = self.memory_usage(unit="GB")
        rows = " rows"
        if self._size == 1:
            rows = " row"
        return "DataFrame([" + keystr + "], {:,}".format(self._size) + rows + ", " + str(mem) + ")"

    def _get_head_tail(self):
        if self._empty:
            return pd.DataFrame()
        self.update_size()
        maxrows = pd.get_option("display.max_rows")
        if self._size <= maxrows:
            newdf = DataFrame()
            for col in self._columns:
                if isinstance(self[col], Categorical):
                    newdf[col] = self[col].categories[self[col].codes]
                else:
                    newdf[col] = self[col]
            newdf._set_index(self.index)
            return newdf.to_pandas(retain_index=True)
        # Being 1 above the threshold causes the PANDAS formatter to split the data frame vertically
        idx = array(list(range(maxrows // 2 + 1)) + list(range(self._size - (maxrows // 2), self._size)))
        newdf = DataFrame()
        for col in self._columns:
            if isinstance(self[col], Categorical):
                newdf[col] = self[col].categories[self[col].codes[idx]]
            else:
                newdf[col] = self[col][idx]
        newdf._set_index(self.index.index[idx])
        return newdf.to_pandas(retain_index=True)

    def _get_head_tail_server(self):
        if self._empty:
            return pd.DataFrame()
        self.update_size()
        maxrows = pd.get_option("display.max_rows")
        if self._size <= maxrows:
            newdf = DataFrame()
            for col in self._columns:
                if isinstance(self[col], Categorical):
                    newdf[col] = self[col].categories[self[col].codes]
                else:
                    newdf[col] = self[col]
            newdf._set_index(self.index)
            return newdf.to_pandas(retain_index=True)
        # Being 1 above the threshold causes the PANDAS formatter to split the data frame vertically
        idx = array(list(range(maxrows // 2 + 1)) + list(range(self._size - (maxrows // 2), self._size)))
        msg_list = []
        for col in self._columns:
            if isinstance(self[col], Categorical):
                msg_list.append(f"Categorical+{col}+{self[col].codes.name}+{self[col].categories.name}")
            elif isinstance(self[col], SegArray):
                msg_list.append(f"SegArray+{col}+{self[col].segments.name}+{self[col].values.name}")
            elif isinstance(self[col], Strings):
                msg_list.append(f"Strings+{col}+{self[col].name}")
            elif isinstance(self[col], Fields):
                msg_list.append(f"Fields+{col}+{self[col].name}")
            elif isinstance(self[col], IPv4):
                msg_list.append(f"IPv4+{col}+{self[col].name}")
            elif isinstance(self[col], Datetime):
                msg_list.append(f"Datetime+{col}+{self[col].name}")
            elif isinstance(self[col], BitVector):
                msg_list.append(f"BitVector+{col}+{self[col].name}")
            else:
                msg_list.append(f"pdarray+{col}+{self[col].name}")

        repMsg = cast(
            str,
            generic_msg(
                cmd="dataframe_idx",
                args={
                    "size": len(msg_list),
                    "idx_name": idx.name,
                    "columns": msg_list,
                },
            ),
        )
        msgList = json.loads(repMsg)

        df_dict = {}
        for m in msgList:
            # Split to [datatype, column, create]
            msg = m.split("+", 2)
            t = msg[0]
            if t == "Strings":
                # Categorical is returned as a strings by indexing categories[codes[idx]]
                df_dict[msg[1]] = Strings.from_return_msg(msg[2])
            elif t == "SegArray":
                # split creates for segments and values
                eles = msg[2].split("+")
                df_dict[msg[1]] = SegArray(create_pdarray(eles[0]), create_pdarray(eles[1]))
            elif t == "Fields":
                df_dict[msg[1]] = Fields(
                    create_pdarray(msg[2]),
                    self[msg[1]].names,
                    MSB_left=self[msg[1]].MSB_left,
                    pad=self[msg[1]].padchar,
                    separator=self[msg[1]].separator,
                    show_int=self[msg[1]].show_int,
                )
            elif t == "IPv4":
                df_dict[msg[1]] = IPv4(create_pdarray(msg[2]))
            elif t == "Datetime":
                df_dict[msg[1]] = Datetime(create_pdarray(msg[2]))
            elif t == "BitVector":
                df_dict[msg[1]] = BitVector(
                    create_pdarray(msg[2]),
                    width=self[msg[1]].width,
                    reverse=self[msg[1]].reverse,
                )
            else:
                df_dict[msg[1]] = create_pdarray(msg[2])

        new_df = DataFrame(df_dict)
        new_df._set_index(self.index.index[idx])
        return new_df.to_pandas(retain_index=True)[self._columns]

    def transfer(self, hostname, port):
        """
        Sends a DataFrame to a different Arkouda server

        Parameters
        ----------
        hostname : str
            The hostname where the Arkouda server intended to
            receive the DataFrame is running.
        port : int_scalars
            The port to send the array over. This needs to be an
            open port (i.e., not one that the Arkouda server is
            running on). This will open up `numLocales` ports,
            each of which in succession, so will use ports of the
            range {port..(port+numLocales)} (e.g., running an
            Arkouda server of 4 nodes, port 1234 is passed as
            `port`, Arkouda will use ports 1234, 1235, 1236,
            and 1237 to send the array data).
            This port much match the port passed to the call to
            `ak.receive_array()`.


        Returns
        -------
        A message indicating a complete transfer

        Raises
        ------
        ValueError
            Raised if the op is not within the pdarray.BinOps set
        TypeError
            Raised if other is not a pdarray or the pdarray.dtype is not
            a supported dtype
        """
        self.update_size()
        idx = self._index
        msg_list = []
        for col in self._columns:
            if isinstance(self[col], Categorical):
                msg_list.append(
                    f"Categorical+{col}+{self[col].codes.name} \
                +{self[col].categories.name}+{self[col]._akNAcode.name}"
                )
            elif isinstance(self[col], SegArray):
                msg_list.append(f"SegArray+{col}+{self[col].segments.name}+{self[col].values.name}")
            elif isinstance(self[col], Strings):
                msg_list.append(f"Strings+{col}+{self[col].name}")
            elif isinstance(self[col], Fields):
                msg_list.append(f"Fields+{col}+{self[col].name}")
            elif isinstance(self[col], IPv4):
                msg_list.append(f"IPv4+{col}+{self[col].name}")
            elif isinstance(self[col], Datetime):
                msg_list.append(f"Datetime+{col}+{self[col].name}")
            elif isinstance(self[col], BitVector):
                msg_list.append(f"BitVector+{col}+{self[col].name}")
            else:
                msg_list.append(f"pdarray+{col}+{self[col].name}")

        repMsg = cast(
            str,
            generic_msg(
                cmd="sendDataframe",
                args={
                    "size": len(msg_list),
                    "idx_name": idx.name,
                    "columns": msg_list,
                    "hostname": hostname,
                    "port": port,
                },
            ),
        )
        return repMsg

    def _shape_str(self):
        return f"{self.size} rows x {self._ncols()} columns"

    def __repr__(self):
        """
        Return ascii-formatted version of the dataframe.
        """

        prt = self._get_head_tail_server()
        with pd.option_context("display.show_dimensions", False):
            retval = prt.__repr__()
        retval += " (" + self._shape_str() + ")"
        return retval

    def _repr_html_(self):
        """
        Return html-formatted version of the dataframe.
        """
        prt = self._get_head_tail_server()

        with pd.option_context("display.show_dimensions", False):
            retval = prt._repr_html_()
        retval += "<p>" + self._shape_str() + "</p>"
        return retval

    def _ipython_key_completions_(self):
        return self._columns

    @classmethod
    def from_pandas(cls, pd_df):
        return DataFrame(initialdata=pd_df)

    def _drop_column(self, keys):
        """
        Drop a column or columns from the dataframe, in-place.

        keys : list
            The labels to be dropped on the given axis
        """
        for key in keys:
            # This will raise an exception if key does not exist
            # Use self.pop(key, None) if we do not want to error
            del self[key]

    def _drop_row(self, keys):
        """
        Drop a row or rows from the dataframe, in-place.

        keys : list
            The indexes to be dropped on the given axis
        """
        idx_list = []
        last_idx = -1
        # sort to ensure we go in ascending order.
        keys.sort()
        for k in keys:
            if not isinstance(k, int):
                raise TypeError("Index keys must be integers.")
            idx_list.append(self.index.index[(last_idx + 1) : k])
            last_idx = k

        idx_list.append(self.index.index[(last_idx + 1) :])

        idx_to_keep = concatenate(idx_list)
        for key in self.keys():
            # using the UserDict.__setitem__ here because we know all the columns are being
            # reset to the same size
            # This avoids the size checks we would do when only setting a single column
            UserDict.__setitem__(self, key, self[key][idx_to_keep])
        self._set_index(idx_to_keep)

    @typechecked
    def drop(
        self,
        keys: Union[str, int, List[Union[str, int]]],
        axis: Union[str, int] = 0,
        inplace: bool = False,
    ) -> Union[None, DataFrame]:
        """
        Drop column/s or row/s from the dataframe.

        Parameters
        ----------
        keys : str, int or list
            The labels to be dropped on the given axis
        axis : int or str
            The axis on which to drop from. 0/'index' - drop rows, 1/'columns' - drop columns
        inplace: bool
            Default False. When True, perform the operation on the calling object.
            When False, return a new object.

        Returns
        -------
            DateFrame when `inplace=False`
            None when `inplace=True`

        Examples
        ----------
        Drop column
        >>> df.drop('col_name', axis=1)

        Drop Row
        >>> df.drop(1)
        or
        >>> df.drop(1, axis=0)
        """

        if isinstance(keys, str) or isinstance(keys, int):
            keys = [keys]

        obj = self if inplace else self.copy()

        if axis == 0 or axis == "index":
            # drop a row
            obj._drop_row(keys)
        elif axis == 1 or axis == "columns":
            # drop column
            obj._drop_column(keys)
        else:
            raise ValueError(f"No axis named {axis} for object type DataFrame")

        # If the dataframe just became empty...
        if len(obj._columns) == 0:
            obj._set_index(None)
            obj._empty = True
        obj.update_size()

        if not inplace:
            return obj

        return None

    def drop_duplicates(self, subset=None, keep="first"):
        """
        Drops duplcated rows and returns resulting DataFrame.

        If a subset of the columns are provided then only one instance of each
        duplicated row will be returned (keep determines which row).

        Parameters
        ----------
        subset : Iterable of column names to use to dedupe.
        keep : {'first', 'last'}, default 'first'
            Determines which duplicates (if any) to keep.

        Returns
        -------
        DataFrame
            DataFrame with duplicates removed.
        """
        if self._empty:
            return self

        if not subset:
            subset = self._columns

        if len(subset) == 1:
            if not subset[0] in self.data:
                raise KeyError(f"{subset[0]} is not a column in the DataFrame.")
            gp = akGroupBy(self.data[subset[0]])

        else:
            for col in subset:
                if col not in self.data:
                    raise KeyError(f"{subset[0]} is not a column in the DataFrame.")

            gp = akGroupBy([self.data[col] for col in subset])

        if keep == "last":
            _segment_ends = concatenate([gp.segments[1:] - 1, array([gp.permutation.size - 1])])
            return self[gp.permutation[_segment_ends]]
        else:
            return self[gp.permutation[gp.segments]]

    @property
    def size(self):
        """
        Returns the number of bytes on the arkouda server.
        """

        self.update_size()
        if self._size is None:
            return 0
        return self._size

    @property
    def dtypes(self):
        dtypes = []
        keys = []
        for key, val in self.items():
            keys.append(key)
            if isinstance(val, pdarray):
                dtypes.append(str(val.dtype))
            elif isinstance(val, Strings):
                dtypes.append("str")
            elif isinstance(val, Categorical):
                dtypes.append("Categorical")
            elif isinstance(val, SegArray):
                dtypes.append("SegArray")
            else:
                raise TypeError(f"Unsupported type encountered for ak.DataFrame, {type(val)}")
        res = Row({key: dtype for key, dtype in zip(keys, dtypes)})
        return res

    @property
    def empty(self):
        return self._empty

    @property
    def shape(self):
        self.update_size()
        num_cols = len(self._columns)
        num_rows = self._size
        return (num_rows, num_cols)

    @property
    def columns(self):
        return self._columns

    @property
    def index(self):
        return self._index

    def _set_index(self, value):
        if isinstance(value, Index) or value is None:
            self._index = value
        elif isinstance(value, (pdarray, Strings)):
            self._index = Index(value)
        elif isinstance(value, list):
            self._index = Index(array(value))
        else:
            raise TypeError(
                f"DataFrame Index can only be constructed from type ak.Index, pdarray or list."
                f" {type(value)} provided."
            )

    @typechecked
    def reset_index(self, size: bool = False, inplace: bool = False) -> Union[None, DataFrame]:
        """
        Set the index to an integer range.

        Useful if this dataframe is the result of a slice operation from
        another dataframe, or if you have permuted the rows and no longer need
        to keep that ordering on the rows.

        Parameters
        ----------
        size : int
            If size is passed, do not attempt to determine size based on
            existing column sizes. Assume caller handles consistency correctly.
        inplace: bool
            Default False. When True, perform the operation on the calling object.
            When False, return a new object.

        Returns
        -------
            DateFrame when `inplace=False`
            None when `inplace=True`

        NOTE
        ----------
        Pandas adds a column 'index' to indicate the original index. Arkouda does not currently
        support this behavior.
        """

        obj = self if inplace else self.copy()

        if not size:
            obj.update_size()
            obj._set_index(arange(obj._size))
        else:
            obj._set_index(arange(size))

        if not inplace:
            return obj
        return None

    @property
    def info(self):
        """
        Returns a summary string of this dataframe.
        """

        self.update_size()

        if self._size is None:
            return "DataFrame([ -- ][ 0 rows : 0 B])"

        keys = [str(key) for key in list(self._columns)]
        keys = [("'" + key + "'") for key in keys]
        keystr = ", ".join(keys)

        # first call to memory_usage() initializes self._bytes
        mem = self.memory_usage()

        # Get units that make the most sense.
        if self._bytes < 1024:
            mem = self.memory_usage(unit="B")
        elif self._bytes < 1024**2:
            mem = self.memory_usage(unit="KB")
        elif self._bytes < 1024**3:
            mem = self.memory_usage(unit="MB")
        else:
            mem = self.memory_usage(unit="GB")
        rows = " rows"
        if self._size == 1:
            rows = " row"
        return "DataFrame([" + keystr + "], {:,}".format(self._size) + rows + ", " + str(mem) + ")"

    def update_size(self):
        """
        Computes the number of bytes on the arkouda server.
        """
        sizes = set()
        for key, val in self.items():
            if val is not None:
                sizes.add(val.size)
        if len(sizes) > 1:
            raise ValueError("Size mismatch in DataFrame columns.")
        if len(sizes) == 0:
            self._size = None
        else:
            self._size = sizes.pop()

    @typechecked
    def _rename_column(
        self, mapper: Union[Callable, Dict], inplace: bool = False
    ) -> Optional[DataFrame]:
        """
        Rename columns within the dataframe

        Parameters
        ----------
        mapper : callable or dict-like
            Function or dictionary mapping existing columns to new columns.
            Nonexistent names will not raise an error.
        inplace: bool
            Default False. When True, perform the operation on the calling object.
            When False, return a new object.
        Returns
        -------
            DateFrame when `inplace=False`
            None when `inplace=True`

        See Also
        -------
        ak.DataFrame._rename_index
        ak.DataFrame.rename
        """
        obj = self if inplace else self.copy()

        if callable(mapper):
            for i in range(0, len(obj._columns)):
                oldname = obj._columns[i]
                newname = mapper(oldname)
                # Only rename if name has changed
                if newname != oldname:
                    obj._columns[i] = newname
                    obj.data[newname] = obj.data[oldname]
                    del obj.data[oldname]
        elif isinstance(mapper, dict):
            for oldname, newname in mapper.items():
                # Only rename if name has changed
                if newname != oldname:
                    try:
                        i = obj._columns.index(oldname)
                        obj._columns[i] = newname
                        obj.data[newname] = obj.data[oldname]
                        del obj.data[oldname]
                    except Exception:
                        pass
        else:
            raise TypeError("Argument must be callable or dict-like")
        if not inplace:
            return obj
        return None

    @typechecked
    def _rename_index(self, mapper: Union[Callable, Dict], inplace: bool = False) -> Optional[DataFrame]:
        """
        Rename indexes within the dataframe

        Parameters
        ----------
        mapper : callable or dict-like
            Function or dictionary mapping existing indexes to new indexes.
            Nonexistent names will not raise an error.
        inplace: bool
            Default False. When True, perform the operation on the calling object.
            When False, return a new object.
        Returns
        -------
            DateFrame when `inplace=False`
            None when `inplace=True`
        See Also
        -------
            ak.DataFrame._rename_column
            ak.DataFrame.rename
        Notes
        -----
            This does not function exactly like pandas. The replacement value here must be
            the same type as the existing value.
        """
        obj = self if inplace else self.copy()
        if callable(mapper):
            for i in range(obj.index.size):
                oldval = obj.index[i]
                newval = mapper(oldval)
                if type(oldval) is not type(newval):
                    raise TypeError("Replacement value must have the same type as the original value")
                obj.index.values[obj.index.values == oldval] = newval
        elif isinstance(mapper, dict):
            for key, val in mapper.items():
                if type(key) is not type(val):
                    raise TypeError("Replacement value must have the same type as the original value")
                obj.index.values[obj.index.values == key] = val
        else:
            raise TypeError("Argument must be callable or dict-like")
        if not inplace:
            return obj
        return None

    @typechecked
    def rename(
        self,
        mapper: Optional[Union[Callable, Dict]] = None,
        index: Optional[Union[Callable, Dict]] = None,
        column: Optional[Union[Callable, Dict]] = None,
        axis: Union[str, int] = 0,
        inplace: bool = False,
    ) -> Optional[DataFrame]:
        """
        Rename indexes or columns according to a mapping.

        Parameters
        ----------
        mapper : callable or dict-like, Optional
            Function or dictionary mapping existing values to new values.
            Nonexistent names will not raise an error.
            Uses the value of axis to determine if renaming column or index
        column : callable or dict-like, Optional
            Function or dictionary mapping existing column names to
            new column names. Nonexistent names will not raise an
            error.
            When this is set, axis is ignored.
        index : callable or dict-like, Optional
            Function or dictionary mapping existing index names to
            new index names. Nonexistent names will not raise an
            error.
            When this is set, axis is ignored
        axis: int or str
            Default 0.
            Indicates which axis to perform the rename.
            0/"index" - Indexes
            1/"column" - Columns
        inplace: bool
            Default False. When True, perform the operation on the calling object.
            When False, return a new object.
        Returns
        -------
            DateFrame when `inplace=False`
            None when `inplace=True`
        Examples
        --------
        >>> df = ak.DataFrame({"A": ak.array([1, 2, 3]), "B": ak.array([4, 5, 6])})
        Rename columns using a mapping
        >>> df.rename(columns={'A':'a', 'B':'c'})
            a   c
        0   1   4
        1   2   5
        2   3   6

        Rename indexes using a mapping
        >>> df.rename(index={0:99, 2:11})
             A   B
        99   1   4
        1   2   5
        11   3   6

        Rename using an axis style parameter
        >>> df.rename(str.lower, axis='column')
            a   b
        0   1   4
        1   2   5
        2   3   6
        """
        if column is not None and index is not None:
            raise RuntimeError("Only column or index can be renamed, cannot rename both at once")

        # convert the axis to the integer value and validate
        if isinstance(axis, str):
            if axis == "column" or axis == "1":
                axis = 1
            elif axis == "index" or axis == "0":
                axis = 0
            else:
                raise ValueError(f"Unknown axis value {axis}. Expecting 0, 1, 'column' or 'index'.")

        if column is not None:
            return self._rename_column(column, inplace)
        elif mapper is not None and axis == 1:
            return self._rename_column(mapper, inplace)
        elif index is not None:
            return self._rename_index(index, inplace)
        elif mapper is not None and axis == 0:
            return self._rename_index(mapper, inplace)
        else:
            raise RuntimeError("Rename expects index or columns to be specified.")

    def append(self, other, ordered=True):
        """
        Concatenate data from 'other' onto the end of this DataFrame, in place.

        Explicitly, use the arkouda concatenate function to append the data
        from each column in other to the end of self. This operation is done
        in place, in the sense that the underlying pdarrays are updated from
        the result of the arkouda concatenate function, rather than returning
        a new DataFrame object containing the result.

        Parameters
        ----------
        other : DataFrame
            The DataFrame object whose data will be appended to this DataFrame.
        ordered: bool
            If False, allow rows to be interleaved for better performance (but
            data within a row remains together). By default, append all rows
            to the end, in input order.

        Returns
        -------
        self
            Appending occurs in-place, but result is returned for compatibility.
        """
        from arkouda.util import generic_concat as util_concatenate

        # Do nothing if the other dataframe is empty
        if other.empty:
            return self

        # Check all the columns to make sure they can be concatenated
        self.update_size()

        keyset = set(self._columns)
        keylist = list(self._columns)

        # Allow for starting with an empty dataframe
        if self.empty:
            self = other.copy()
        # Keys don't match
        elif keyset != set(other._columns):
            raise KeyError("Key mismatch; keys must be identical in both DataFrames.")
        # Keys do match
        else:
            tmp_data = {}
            for key in keylist:
                try:
                    tmp_data[key] = util_concatenate([self[key], other[key]], ordered=ordered)
                except TypeError as e:
                    raise TypeError(
                        f"Incompatible types for column {key}: {type(self[key])} vs {type(other[key])}"
                    ) from e
            self.data = tmp_data

        # Clean up
        self.reset_index(inplace=True)
        self.update_size()
        self._empty = False
        return self

    @classmethod
    def concat(cls, items, ordered=True):
        """
        Essentially an append, but diffenent formatting
        """
        from arkouda.util import generic_concat as util_concatenate

        if len(items) == 0:
            return cls()
        first = True
        columnset = set()
        columnlist = []
        for df in items:
            # Allow for an empty dataframe
            if df.empty:
                continue
            if first:
                columnset = set(df._columns)
                columnlist = df._columns
                first = False
            else:
                if set(df._columns) != columnset:
                    raise KeyError("Cannot concatenate DataFrames with mismatched columns")
        # if here, columns match
        ret = cls()
        for col in columnlist:
            try:
                ret[col] = util_concatenate([df[col] for df in items], ordered=ordered)
            except TypeError:
                raise TypeError(f"Incompatible types for column {col}")
        return ret

    def head(self, n=5):
        """
        Return the first `n` rows.

        This function returns the first `n` rows of the the dataframe. It is
        useful for quickly verifying data, for example, after sorting or
        appending rows.

        Parameters
        ----------
        n : int
            Number of rows to select.

        Returns
        -------
        ak.DataFrame
            The first `n` rows of the DataFrame.

        See Also
        --------
        tail
        """

        return self[:n]

    def tail(self, n=5):
        """
        Return the last `n` rows.

        This function returns the last `n` rows for the dataframe. It is
        useful for quickly testing if your object has the right type of data in
        it.

        Parameters
        ----------
        n : int (default=5)
            Number of rows to select.

        Returns
        -------
        ak.DataFrame
            The last `n` rows of the DataFrame.

        See Also
        --------
        ak.dataframe.head
        """

        self.update_size()
        if self._size <= n:
            return self
        return self[self._size - n :]

    def sample(self, n=5):
        """
        Return a random sample of `n` rows.

        Parameters
        ----------
        n : int (default=5)
            Number of rows to return.

        Returns
        -------
        ak.DataFrame
            The sampled `n` rows of the DataFrame.
        """
        self.update_size()
        if self._size <= n:
            return self
        return self[array(random.sample(range(self._size), n))]

    def GroupBy(self, keys, use_series=False, as_index=True, dropna=True):
        """
        Group the dataframe by a column or a list of columns.

        Parameters
        ----------
        keys : string or list
            An (ordered) list of column names or a single string to group by.
        use_series : bool (default=False)
            If True, returns an arkouda.dataframe.GroupBy object.
            Otherwise an arkouda.groupbyclass.GroupBy object.
        as_index: bool (default=True)
            If True, groupby columns will be set as index
            otherwise, the groupby columns will be treated as DataFrame columns.
        dropna : bool (default=True)
            If True, and the groupby keys contain NaN values,
            the NaN values together with the corresponding row will be dropped.
            Otherwise, the rows corresponding to NaN values will be kept.
        Returns
        -------
        GroupBy
            Either an ak arkouda.groupbyclass.GroupBy or an arkouda.dataframe.GroupBy object.

        See Also
        --------
        arkouda.GroupBy
        """

        self.update_size()
        if isinstance(keys, str):
            cols = self.data[keys]
        elif not isinstance(keys, (list, tuple)):
            raise TypeError("keys must be a column name or a list/tuple of column names")
        elif len(keys) == 1:
            cols = self.data[keys[0]]
        else:
            cols = [self.data[col] for col in keys]

        gb = akGroupBy(cols, dropna=dropna)
        if use_series:
            gb = GroupBy(gb, self, gb_key_names=keys, as_index=as_index)
        return gb

    def memory_usage(self, unit="GB"):
        """
        Print the size of this DataFrame.

        Parameters
        ----------
        unit : str
            Unit to return. One of {'KB', 'MB', 'GB'}.

        Returns
        -------
        int
            The number of bytes used by this DataFrame in [unit]s.
        """

        KB = 1024
        MB = KB * KB
        GB = MB * KB
        self._bytes = 0
        for key, val in self.items():
            if isinstance(val, pdarray):
                self._bytes += (val.dtype).itemsize * val.size
            elif isinstance(val, Strings):
                self._bytes += val.nbytes
        if unit == "B":
            return "{:} B".format(int(self._bytes))
        elif unit == "MB":
            return "{:} MB".format(int(self._bytes / MB))
        elif unit == "KB":
            return "{:} KB".format(int(self._bytes / KB))
        return "{:.2f} GB".format(self._bytes / GB)

    def to_pandas(self, datalimit=maxTransferBytes, retain_index=False):
        """
        Send this DataFrame to a pandas DataFrame.

        Parameters
        ----------
        datalimit : int (default=arkouda.client.maxTransferBytes)
            The maximum number size, in megabytes to transfer. The requested
            DataFrame will be converted to a pandas DataFrame only if the
            estimated size of the DataFrame does not exceed this value.

        retain_index : book (default=False)
            Normally, to_pandas() creates a new range index object. If you want
            to keep the index column, set this to True.

        Returns
        -------
        pandas.DataFrame
            The result of converting this DataFrame to a pandas DataFrame.
        """

        self.update_size()

        # Estimate how much memory would be required for this DataFrame
        nbytes = 0
        for key, val in self.items():
            if isinstance(val, pdarray):
                nbytes += (val.dtype).itemsize * self._size
            elif isinstance(val, Strings):
                nbytes += val.nbytes

        KB = 1024
        MB = KB * KB
        GB = MB * KB

        # Get units that make the most sense.
        msg = ""
        if nbytes < KB:
            msg = "{:,} B".format(nbytes)
        elif nbytes < MB:
            msg = "{:,} KB".format(int(nbytes / KB))
        elif nbytes < GB:
            msg = "{:,} MB".format(int(nbytes / MB))
            print(f"This transfer will use {msg} .")
        else:
            msg = "{:,} GB".format(int(nbytes / GB))
            print(f"This will transfer {msg} from arkouda to pandas.")
        # If the total memory transfer requires more than `datalimit` per
        # column, we will warn the user and return.
        if nbytes > (datalimit * len(self._columns) * MB):
            msg = f"This operation would transfer more than {datalimit} bytes."
            warn(msg, UserWarning)
            return None

        # Proceed with conversion if possible
        pandas_data = {}
        for key in self._columns:
            val = self[key]
            try:
                # in order for proper pandas functionality, SegArrays must be seen as 1d
                # and therefore need to be converted to list
                pandas_data[key] = val.to_ndarray() if not isinstance(val, SegArray) else val.to_list()
            except TypeError:
                raise IndexError("Bad index type or format.")

        # Return a new dataframe with original indices if requested.
        if retain_index and self.index is not None:
            index = self.index.to_pandas()
            return pd.DataFrame(data=pandas_data, index=index)
        else:
            return pd.DataFrame(data=pandas_data)

    def _prep_data(self, index=False, columns=None):
        # if no columns are stored, we will save all columns
        if columns is None:
            data = self.data
        else:
            data = {c: self.data[c] for c in columns}

        if index:
            data["Index"] = self.index.values
        return data

    def to_hdf(self, path, index=False, columns=None, file_type="distribute"):
        """
        Save DataFrame to disk as hdf5, preserving column names.

        Parameters
        ----------
        path : str
            File path to save data
        index : bool
            If True, save the index column. By default, do not save the index.
        columns: List
            List of columns to include in the file. If None, writes out all columns
        file_type: str (single | distribute)
            Default: distribute
            Whether to save to a single file or distribute across Locales
        Returns
        -------
        None
        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray
        Notes
        -----
        This method saves one file per locale of the arkouda server. All
        files are prefixed by the path argument and suffixed by their
        locale number.
        See Also
        ---------
        to_parquet, load
        """
        from arkouda.io import to_hdf

        data = self._prep_data(index=index, columns=columns)
        to_hdf(data, prefix_path=path, file_type=file_type)

    def _to_hdf_snapshot(self, path, dataset="DataFrame", mode="truncate", file_type="distribute"):
        """
        Save a dataframe as a group with columns within the group. This allows saving other
        datasets in the HDF5 file without impacting the integrity of the dataframe
        This is only used for the snapshot workflow
        Parameters
        ----------
        path : str
            File path to save data
        dataset: str
            Name to save the dataframe under within the file
            Only used when as_dataset=True
        mode: str (truncate | append)
            Default: trunate
            Indicates whether the dataset should truncate the file and write or append
            to the file
            Only used when as_dataset=True
        file_type: str (single | distribute)
            Default: distribute
            Whether to save to a single file or distribute across Locales
            Only used when as_dataset=True

        Returns
        -------
        None
        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray
        """
        from arkouda.categorical import Categorical as Categorical_
        from arkouda.io import _file_type_to_int, _mode_str_to_int

        column_data = [
            obj.name
            if not isinstance(obj, (Categorical_, SegArray))
            else json.dumps(
                {
                    "codes": obj.codes.name,
                    "categories": obj.categories.name,
                    "NA_codes": obj._akNAcode.name,
                    **({"permutation": obj.permutation.name} if obj.permutation is not None else {}),
                    **({"segments": obj.segments.name} if obj.segments is not None else {}),
                }
            )
            if isinstance(obj, Categorical_)
            else json.dumps({"segments": obj.segments.name, "values": obj.values.name})
            for k, obj in self.items()
        ]
        dtypes = [
            str(obj.categories.dtype) if isinstance(obj, Categorical_) else str(obj.dtype)
            for obj in self.values()
        ]
        col_objTypes = [
            obj.special_objType if hasattr(obj, "special_objType") else obj.objType
            for obj in self.values()
        ]
        return cast(
            str,
            generic_msg(
                cmd="tohdf",
                args={
                    "filename": path,
                    "dset": dataset,
                    "file_format": _file_type_to_int(file_type),
                    "write_mode": _mode_str_to_int(mode),
                    "objType": self.objType,
                    "num_cols": len(self.columns),
                    "column_names": self.columns,
                    "column_objTypes": col_objTypes,
                    "column_dtypes": dtypes,
                    "columns": column_data,
                    "index": self.index.values.name,
                },
            ),
        )

    def update_hdf(self, prefix_path: str, index=False, columns=None, repack: bool = True):
        """
        Overwrite the dataset with the name provided with this dataframe. If
        the dataset does not exist it is added

        Parameters
        -----------
        prefix_path : str
            Directory and filename prefix that all output files share
        index : bool
            If True, save the index column. By default, do not save the index.
        columns: List
            List of columns to include in the file. If None, writes out all columns
        repack: bool
            Default: True
            HDF5 does not release memory on delete. When True, the inaccessible
            data (that was overwritten) is removed. When False, the data remains, but is
            inaccessible. Setting to false will yield better performance, but will cause
            file sizes to expand.

        Returns
        --------
        str - success message if successful

        Raises
        -------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray

        Notes
        ------
        - If file does not contain File_Format attribute to indicate how it was saved,
          the file name is checked for _LOCALE#### to determine if it is distributed.
        - If the dataset provided does not exist, it will be added
        """
        from arkouda.io import update_hdf

        data = self._prep_data(index=index, columns=columns)
        update_hdf(data, prefix_path=prefix_path, repack=repack)

    def to_parquet(
        self,
        path,
        index=False,
        columns=None,
        compression: Optional[str] = None,
        convert_categoricals: bool = False,
    ):
        """
        Save DataFrame to disk as parquet, preserving column names.

        Parameters
        ----------
        path : str
            File path to save data
        index : bool
            If True, save the index column. By default, do not save the index.
        columns: List
            List of columns to include in the file. If None, writes out all columns
        compression : str (Optional)
            Default None
            Provide the compression type to use when writing the file.
            Supported values: snappy, gzip, brotli, zstd, lz4
        convert_categoricals: bool
            Defaults to False
            Parquet requires all columns to be the same size and Categoricals
            don't satisfy that requirement.
            if set, write the equivalent Strings in place of any Categorical columns.
        Returns
        -------
        None
        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray
        Notes
        -----
        This method saves one file per locale of the arkouda server. All
        files are prefixed by the path argument and suffixed by their
        locale number.
        See Also
        ---------
        to_hdf, load
        """
        from arkouda.io import to_parquet

        data = self._prep_data(index=index, columns=columns)
        if not convert_categoricals and any(isinstance(val, Categorical) for val in data.values()):
            raise ValueError(
                "to_parquet doesn't support Categorical columns. To write the equivalent "
                "Strings in place of any Categorical columns, rerun with convert_categoricals "
                "set to True."
            )
        to_parquet(
            data,
            prefix_path=path,
            compression=compression,
            convert_categoricals=convert_categoricals,
        )

    @typechecked
    def to_csv(
        self,
        path: str,
        index: bool = False,
        columns: Optional[List[str]] = None,
        col_delim: str = ",",
        overwrite: bool = False,
    ):
        """
        Writes DataFrame to CSV file(s). File will contain a column for each column in the DataFrame.
        All CSV Files written by Arkouda include a header denoting data types of the columns.
        Unlike other file formats, CSV files store Strings as their UTF-8 format instead of storing
        bytes as uint(8).

        Parameters
        -----------
        path: str
            The filename prefix to be used for saving files. Files will have _LOCALE#### appended
            when they are written to disk.
        index: bool
            Defaults to False. If True, the index of the DataFrame will be written to the file
            as a column
        columns: List[str] (Optional)
            Column names to assign when writing data
        col_delim: str
            Defaults to ",". Value to be used to separate columns within the file.
            Please be sure that the value used DOES NOT appear in your dataset.
        overwrite: bool
            Defaults to False. If True, any existing files matching your provided prefix_path will
            be overwritten. If False, an error will be returned if existing files are found.

        Returns
        --------
        None

        Raises
        ------
        ValueError
            Raised if all datasets are not present in all parquet files or if one or
            more of the specified files do not exist
        RuntimeError
            Raised if one or more of the specified files cannot be opened.
            If `allow_errors` is true this may be raised if no values are returned
            from the server.
        TypeError
            Raised if we receive an unknown arkouda_type returned from the server

        Notes
        ------
        - CSV format is not currently supported by load/load_all operations
        - The column delimiter is expected to be the same for column names and data
        - Be sure that column delimiters are not found within your data.
        - All CSV files must delimit rows using newline (`\n`) at this time.
        """
        from arkouda.io import to_csv

        data = self._prep_data(index=index, columns=columns)
        to_csv(data, path, names=columns, col_delim=col_delim, overwrite=overwrite)

    @classmethod
    def read_csv(cls, filename: str, col_delim: str = ","):
        """
        Read the columns of a CSV file into an Arkouda DataFrame.
        If the file contains the appropriately formatted header, typed data will be returned.
        Otherwise, all data will be returned as a Strings objects.

        Parameters
        -----------
        filename: str
            Filename to read data from
        col_delim: str
            Defaults to ",". The delimiter for columns within the data.

        Returns
        --------
        Arkouda DataFrame containing the columns from the CSV file.

        Raises
        ------
        ValueError
            Raised if all datasets are not present in all parquet files or if one or
            more of the specified files do not exist
        RuntimeError
            Raised if one or more of the specified files cannot be opened.
            If `allow_errors` is true this may be raised if no values are returned
            from the server.
        TypeError
            Raised if we receive an unknown arkouda_type returned from the server

        See Also
        ---------
        to_csv

        Notes
        ------
        - CSV format is not currently supported by load/load_all operations
        - The column delimiter is expected to be the same for column names and data
        - Be sure that column delimiters are not found within your data.
        - All CSV files must delimit rows using newline (`\n`) at this time.
        - Unlike other file formats, CSV files store Strings as their UTF-8 format instead of storing
        bytes as uint(8).

        """
        from arkouda.io import read_csv

        data = read_csv(filename, column_delim=col_delim)
        return cls(data)

    def save(
        self,
        path,
        index=False,
        columns=None,
        file_format="HDF5",
        file_type="distribute",
        compression: Optional[str] = None,
    ):
        """
        DEPRECATED
        Save DataFrame to disk, preserving column names.
        Parameters
        ----------
        path : str
            File path to save data
        index : bool
            If True, save the index column. By default, do not save the index.
        columns: List
            List of columns to include in the file. If None, writes out all columns
        file_format: str
            'HDF5' or 'Parquet'. Defaults to 'HDF5'
        file_type: str
            ("single" | "distribute")
            Defaults to distribute.
            If single, will right a single file to locale 0
        compression: str (Optional)
            (None | "snappy" | "gzip" | "brotli" | "zstd" | "lz4")
            Compression type. Only used for Parquet
        Notes
        -----
        This method saves one file per locale of the arkouda server. All
        files are prefixed by the path argument and suffixed by their
        locale number.
        See Also
        --------
        to_parquet, to_hdf
        """
        warn(
            "ak.DataFrame.save has been deprecated. "
            "Please use ak.DataFrame.to_hdf or ak.DataFrame.to_parquet",
            DeprecationWarning,
        )

        if file_format.lower() == "hdf5":
            return self.to_hdf(path, index=index, columns=columns, file_type=file_type)
        elif file_format.lower() == "parquet":
            return self.to_parquet(path, index=index, columns=columns, compression=compression)
        else:
            raise ValueError("Valid file types are HDF5 or Parquet")

    @classmethod
    def load(cls, prefix_path, file_format="INFER"):
        """
        Load dataframe from file
        file_format needed for consistency with other load functions
        """
        from arkouda.io import (
            _dict_recombine_segarrays_categoricals,
            get_filetype,
            load_all,
        )

        prefix, extension = os.path.splitext(prefix_path)
        first_file = f"{prefix}_LOCALE0000{extension}"
        filetype = get_filetype(first_file) if file_format.lower() == "infer" else file_format

        # columns load backwards
        df = cls(_dict_recombine_segarrays_categoricals(load_all(prefix_path, file_format=filetype)))
        # if parquet, return reversed dataframe to match what was saved
        return df if filetype == "HDF5" else df[df.columns[::-1]]

    def argsort(self, key, ascending=True):
        """
        Return the permutation that sorts the dataframe by `key`.

        Parameters
        ----------
        key : str
            The key to sort on.

        Returns
        -------
        ak.pdarray
            The permutation array that sorts the data on `key`.
        """

        if self._empty:
            return array([], dtype=akint64)
        if ascending:
            return argsort(self[key])
        else:
            if isinstance(self[key], pdarray) and self[key].dtype in (
                akint64,
                akfloat64,
            ):
                return argsort(-self[key])
            else:
                return argsort(self[key])[arange(self.size - 1, -1, -1)]

    def coargsort(self, keys, ascending=True):
        """
        Return the permutation that sorts the dataframe by `keys`.

        Sorting using Strings may not yield correct results

        Parameters
        ----------
        keys : list
            The keys to sort on.

        Returns
        -------
        ak.pdarray
            The permutation array that sorts the data on `keys`.
        """

        if self._empty:
            return array([], dtype=akint64)
        arrays = []
        for key in keys:
            arrays.append(self[key])
        i = coargsort(arrays)
        if not ascending:
            i = i[arange(self.size - 1, -1, -1)]
        return i

    def __reindex(self, idx):
        if isinstance(self.index, MultiIndex):
            new_index = MultiIndex(self.index[idx].values, name=self.index.name, names=self.index.names)
        elif isinstance(self.index, Index) and hasattr(self.index, "name"):
            new_index = Index(self.index[idx], name=self.index.name)
        else:
            new_index = Index(self.index[idx])

        return DataFrame(self[idx], index=new_index)

    def sort_index(self, ascending=True):
        """
        Sort the DataFrame by indexed columns.

        Note: Fails on sorting ak.Strings when multiple columns being sorted

        Parameters
        ----------
        ascending : bool
            Sort values in ascending (default) or descending order.

        """

        idx = self.index.argsort(ascending=ascending)

        return self.__reindex(idx)

    def sort_values(self, by=None, ascending=True):
        """
        Sort the DataFrame by one or more columns.

        If no column is specified, all columns are used.

        Note: Fails on sorting ak.Strings when multiple columns being sorted

        Parameters
        ----------
        by : str or list/tuple of str
            The name(s) of the column(s) to sort by.
        ascending : bool
            Sort values in ascending (default) or descending order.

        See Also
        --------
        apply_permutation, sorted
        """

        if self._empty:
            return array([], dtype=akint64)
        if by is None:
            if len(self._columns) == 1:
                i = self.argsort(self._columns[0], ascending=ascending)
            else:
                i = self.coargsort(self._columns, ascending=ascending)
        elif isinstance(by, str):
            i = self.argsort(by, ascending=ascending)
        elif isinstance(by, (list, tuple)):
            i = self.coargsort(by, ascending=ascending)
        else:
            raise TypeError("Column name(s) must be str or list/tuple of str")
        return self[i]

    def apply_permutation(self, perm):
        """
        Apply a permutation to an entire DataFrame.

        This may be useful if you want to unsort an DataFrame, or even to
        apply an arbitrary permutation such as the inverse of a sorting
        permutation.

        Parameters
        ----------
        perm : ak.pdarray
            A permutation array. Should be the same size as the data
            arrays, and should consist of the integers [0,size-1] in
            some order. Very minimal testing is done to ensure this
            is a permutation.

        See Also
        --------
        sort
        """

        if (perm.min() != 0) or (perm.max() != perm.size - 1):
            raise ValueError("The indicated permutation is invalid.")
        if unique(perm).size != perm.size:
            raise ValueError("The indicated permutation is invalid.")
        for key, val in self.data.items():
            self[key] = self[key][perm]
        self._set_index(self.index[perm])

    def filter_by_range(self, keys, low=1, high=None):
        """
        Find all rows where the value count of the items in a given set of
        columns (keys) is within the range [low, high].

        To filter by a specific value, set low == high.

        Parameters
        ----------
        keys : list or str
            The names of the columns to group by
        low : int (default=1)
            The lowest value count.
        high : int (default=None)
            The highest value count, default to unlimited.

        Returns
        -------
        pdarray
            An array of boolean values for qualified rows in this DataFrame.

        See Also
        --------
        filter_by_count
        """

        if isinstance(keys, str):
            keys = [keys]
        gb = self.GroupBy(keys, use_series=False)
        vals, cts = gb.count()
        if not high:
            positions = where(cts >= low, 1, 0)
        else:
            positions = where(((cts >= low) & (cts <= high)), 1, 0)

        broadcast = gb.broadcast(positions, permute=False)
        broadcast = broadcast == 1
        return broadcast[invert_permutation(gb.permutation)]

    def copy(self, deep=True):
        """
        Make a copy of this object's data.

        When `deep = True` (default), a new object will be created with a copy of
        the calling object's data. Modifications to the data of the copy will not
        be reflected in the original object.


        When `deep = False` a new object will be created without copying the
        calling object's data. Any changes to the data of the original object will
        be reflected in the shallow copy, and vice versa.

        Parameters
        ----------
        deep : bool (default=True)
            When True, return a deep copy. Otherwise, return a shallow copy.

        Returns
        -------
        aku.DataFrame
            A deep or shallow copy according to caller specification.
        """

        if deep:
            res = DataFrame()
            res._size = self._size
            res._bytes = self._bytes
            res._empty = self._empty
            res._columns = self._columns[:]  # if this is not a slice, droping columns modifies both

            for key, val in self.items():
                res[key] = val[:]

            # if this is not a slice, renaming indexes with update both
            res._set_index(Index(self.index.index[:]))

            return res
        else:
            return DataFrame(self)

    def groupby(self, keys, use_series=True, as_index=True, dropna=True):
        """Group the dataframe by a column or a list of columns.  Alias for GroupBy

        Parameters
        ----------
        keys : str or list(str)
            a single column name or a list of column names
        use_series : bool (default = True)
            Change return type to Arkouda Groupby object.
        as_index: bool (default=True)
            If true groupby aggregation values will be treated as an index.
            Otherwise, the groupby values will be treated as DataFrame column(s).
        dropna : bool (default=True)
            If True, and the groupby keys contain NaN values,
            the NaN values together with the corresponding row will be dropped.
            Otherwise, the rows corresponding to NaN values will be kept.
        Returns
        -------
        An arkouda Groupby instance
        """

        return self.GroupBy(keys, use_series, as_index=as_index, dropna=dropna)

    @typechecked
    def isin(self, values: Union[pdarray, Dict, Series, DataFrame]) -> DataFrame:
        """
        Determine whether each element in the DataFrame is contained in values.

        Parameters
        __________
        values : pdarray, dict, Series, or DataFrame
            The values to check for in DataFrame. Series can only have a single index.

        Returns
        _______
        DataFrame
            Arkouda DataFrame of booleans showing whether each element in the DataFrame is
            contained in values

        See Also
        ________
        ak.Series.isin

        Notes
        _____
        - Pandas supports values being an iterable type. In arkouda, we replace this with pdarray
        - Pandas supports ~ operations. Currently, ak.DataFrame does not support this.

        Examples
        ________
        >>> df = ak.DataFrame({'col_A': ak.array([7, 3]), 'col_B':ak.array([1, 9])})
        >>> df
            col_A  col_B
        0      7      1
        1      3      9 (2 rows x 2 columns)

        When `values` is a pdarray, check every value in the DataFrame to determine if
        it exists in values
        >>> df.isin(ak.array([0, 1]))
           col_A  col_B
        0  False   True
        1  False  False (2 rows x 2 columns)

        When `values` is a dict, the values in the dict are passed to check the column
        indicated by the key
        >>> df.isin({'col_A': ak.array([0, 3])})
            col_A  col_B
        0  False  False
        1   True  False (2 rows x 2 columns)

        When `values` is a Series, each column is checked if values is present positionally.
        This means that for `True` to be returned, the indexes must be the same.
        >>> i = ak.Index(ak.arange(2))
        >>> s = ak.Series(data=[3, 9], index=i)
        >>> df.isin(s)
            col_A  col_B
        0  False  False
        1  False   True (2 rows x 2 columns)

        When `values` is a DataFrame, the index and column must match.
        Note that 9 is not found because the column name does not match.
        >>> other_df = ak.DataFrame({'col_A':ak.array([7, 3]), 'col_C':ak.array([0, 9])})
        >>> df.isin(other_df)
            col_A  col_B
        0   True  False
        1   True  False (2 rows x 2 columns)
        """
        if isinstance(values, pdarray):
            # flatten the DataFrame so single in1d can be used.
            flat_in1d = in1d(concatenate(list(self.data.values())), values)
            segs = concatenate(
                [
                    array([0]),
                    cumsum(array([self.data[col].size for col in self.columns])),
                ]
            )
            df_def = {col: flat_in1d[segs[i] : segs[i + 1]] for i, col in enumerate(self.columns)}
        elif isinstance(values, Dict):
            # key is column name, val is the list of values to check
            df_def = {
                col: (
                    in1d(self.data[col], values[col])
                    if col in values.keys()
                    else zeros(self.size, dtype=akbool)
                )
                for col in self.columns
            }
        elif isinstance(values, DataFrame) or (
            isinstance(values, Series) and isinstance(values.index, Index)
        ):
            # create the dataframe with all false
            df_def = {col: zeros(self.size, dtype=akbool) for col in self.columns}
            # identify the indexes in both
            rows_self, rows_val = intersect(self.index.index, values.index.index, unique=True)

            # used to sort the rows with only the indexes in both
            sort_self = self.index[rows_self].argsort()
            sort_val = values.index[rows_val].argsort()
            # update values in columns that exist in both. only update the rows whose indexes match

            for col in self.columns:
                if isinstance(values, DataFrame) and col in values.columns:
                    df_def[col][rows_self] = (
                        self.data[col][rows_self][sort_self] == values.data[col][rows_val][sort_val]
                    )
                elif isinstance(values, Series):
                    df_def[col][rows_self] = (
                        self.data[col][rows_self][sort_self] == values.values[rows_val][sort_val]
                    )
        else:
            # pandas provides the same error in this case
            raise ValueError("Cannot compute isin with duplicate axis.")

        return DataFrame(df_def, index=self.index)

    def corr(self) -> DataFrame:
        """
        Return new DataFrame with pairwise correlation of columns

        Returns
        -------
        DataFrame
            Arkouda DataFrame containing correlation matrix of all columns

        Raises
        ------
        RuntimeError
            Raised if there's a server-side error thrown

        See Also
        --------
        pdarray.corr

        Notes
        -----
        Generates the correlation matrix using Pearson R for all columns

        Attempts to convert to numeric values where possible for inclusion in the matrix.
        """

        def numeric_help(d):
            if isinstance(d, Strings):
                d = Categorical(d)
            return d if isinstance(d, pdarray) else d.codes

        args = {
            "size": len(self.columns),
            "columns": self.columns,
            "data_names": [numeric_help(self[c]) for c in self.columns],
        }

        ret_dict = json.loads(generic_msg(cmd="corrMatrix", args=args))
        return DataFrame({c: create_pdarray(ret_dict[c]) for c in self.columns})

    @typechecked
    def merge(
        self,
        right: DataFrame,
        on: Optional[Union[str, List[str]]] = None,
        how: str = "inner",
        left_suffix: str = "_x",
        right_suffix: str = "_y",
    ) -> DataFrame:
        """
        Utilizes the ak.join.inner_join_merge and the ak.join.right_join_merge
        functions to return a merged Arkouda DataFrame object
        containing rows from both DataFrames as specified by the merge
        condition (based on the "how" and "on" parameters). For this function self
        is considered the left dataframe.
        Based on pandas merge functionality.
        https://github.com/pandas-dev/pandas/blob/main/pandas/core/reshape/merge.py#L137

        Parameters
        ----------
        right: DataFrame
            The Right DataFrame to be joined
        on: Optional[Union[str, List[str]]] = None
            The name or list of names of the DataFrame column(s) to join on.
            If on is None, this defaults to the intersection of the columns in both DataFrames.
        how:  str = "inner",
            The merge condition.
            Must be "inner", "left", or "right"
        left_suffix: str = "_x"
            A string indicating the suffix to add to columns from the left dataframe for overlapping
            column names in both left and right. Defaults to "_x". Only used when how is "inner"
        right_suffix: str = "_y"
            A string indicating the suffix to add to columns from the right dataframe for overlapping
            column names in both left and right. Defaults to "_y". Only used when how is "inner"

        Returns
        -------
        DataFrame
            Joined Arkouda DataFrame

        Note: Multiple column joins are only supported for integer columns
        """
        return merge(self, right, on, how, left_suffix, right_suffix)

    @typechecked
    def register(self, user_defined_name: str) -> DataFrame:
        """
        Register this DataFrame object and underlying components with the Arkouda server

        Parameters
        ----------
        user_defined_name : str
            user defined name the DataFrame is to be registered under,
            this will be the root name for underlying components

        Returns
        -------
        DataFrame
            The same DataFrame which is now registered with the arkouda server and has an updated name.
            This is an in-place modification, the original is returned to support a
            fluid programming style.
            Please note you cannot register two different DataFrames with the same name.

        Raises
        ------
        TypeError
            Raised if user_defined_name is not a str
        RegistrationError
            If the server was unable to register the DataFrame with the user_defined_name

        See also
        --------
        unregister, attach, unregister_dataframe_by_name, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.

        Any changes made to a DataFrame object after registering with the server may not be reflected
        in attached copies.
        """
        from arkouda.categorical import Categorical as Categorical_

        if self.registered_name is not None and self.is_registered():
            raise RegistrationError(f"This object is already registered as {self.registered_name}")
        column_data = [
            obj.name
            if not isinstance(obj, (Categorical_, SegArray, BitVector))
            else json.dumps(
                {
                    "codes": obj.codes.name,
                    "categories": obj.categories.name,
                    "NA_codes": obj._akNAcode.name,
                    **({"permutation": obj.permutation.name} if obj.permutation is not None else {}),
                    **({"segments": obj.segments.name} if obj.segments is not None else {}),
                }
            )
            if isinstance(obj, Categorical_)
            else json.dumps({"segments": obj.segments.name, "values": obj.values.name})
            if isinstance(obj, SegArray)
            else json.dumps(
                {
                    "name": obj.name,
                    "width": obj.width,
                    "reverse": obj.reverse,
                }  # BitVector Case
            )
            for obj in self.values()
        ]

        col_objTypes = [
            obj.special_objType if hasattr(obj, "special_objType") else obj.objType
            for obj in self.values()
        ]

        generic_msg(
            cmd="register",
            args={
                "name": user_defined_name,
                "objType": self.objType,
                "idx": self.index.values.name,
                "num_cols": len(self.columns),
                "column_names": self.columns,
                "columns": column_data,
                "col_objTypes": col_objTypes,
            },
        )
        self.registered_name = user_defined_name
        return self

    def unregister(self):
        """
        Unregister this DataFrame object in the arkouda server which was previously
        registered using register() and/or attached to using attach()

        Raises
        ------
        RegistrationError
            If the object is already unregistered or if there is a server error
            when attempting to unregister

        See also
        --------
        register, attach, unregister_dataframe_by_name, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.
        """
        from arkouda.util import unregister

        if not self.registered_name:
            raise RegistrationError("This object is not registered")
        unregister(self.registered_name)
        self.registered_name = None  # Clear our internal DataFrame object name

    def is_registered(self) -> bool:
        """
        Return True if the object is contained in the registry

        Returns
        -------
        bool
            Indicates if the object is contained in the registry

        Raises
        ------
        RegistrationError
            Raised if there's a server-side error or a mismatch of registered components

        See Also
        --------
        register, attach, unregister, unregister_dataframe_by_name

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.
        """
        from arkouda.util import is_registered

        if self.registered_name is None:
            return False  # Dataframe cannot be registered as a component
        return is_registered(self.registered_name)

    @staticmethod
    def attach(user_defined_name: str) -> DataFrame:
        """
        Function to return a DataFrame object attached to the registered name in the
        arkouda server which was registered using register()

        Parameters
        ----------
        user_defined_name : str
            user defined name which DataFrame object was registered under

        Returns
        -------
        DataFrame
               The DataFrame object created by re-attaching to the corresponding server components

        Raises
        ------
        RegistrationError
            if user_defined_name is not registered

        See Also
        --------
        register, is_registered, unregister
        """
        import warnings

        from arkouda.util import attach

        warnings.warn(
            "ak.DataFrame.attach() is deprecated. Please use ak.attach() instead.",
            DeprecationWarning,
        )
        return attach(user_defined_name)

    @staticmethod
    @typechecked
    def unregister_dataframe_by_name(user_defined_name: str) -> None:
        """
        Function to unregister DataFrame object by name which was registered
        with the arkouda server via register()

        Parameters
        ----------
        user_defined_name : str
            Name under which the DataFrame object was registered

        Raises
        -------
        TypeError
            if user_defined_name is not a string
        RegistrationError
            if there is an issue attempting to unregister any underlying components

        See Also
        --------
        register, unregister, attach, is_registered
        """
        import warnings

        from arkouda.util import unregister

        warnings.warn(
            "ak.DataFrame.unregister_dataframe_by_name() is deprecated. "
            "Please use ak.unregister() instead.",
            DeprecationWarning,
        )
        return unregister(user_defined_name)

    @staticmethod
    def _parse_col_name(entryName, dfName):
        """
        Helper method used by from_return_msg to parse the registered name of the data component
        and pull out the column type and column name

        Parameters
        ----------
        entryName : string
            The full registered name of the data component

        dfName : string
            The name of the DataFrame

        Returns
        -------
        Tuple (columnName, columnType)
        """
        nameParts = entryName.split(" ")
        regName = nameParts[1] if len(nameParts) > 1 else nameParts[0]
        colParts = regName.split("_")
        colType = colParts[2]

        # Case of '_' in the column or dataframe name
        if len(colParts) > 5:
            nameInd = regName.rindex(dfName) - 1
            startInd = len(colType) + 9
            return regName[startInd:nameInd], colType
        else:
            return colParts[3], colType

    @classmethod
    def from_return_msg(cls, rep_msg):
        from arkouda.categorical import Categorical as Categorical_

        data = json.loads(rep_msg)
        idx = None
        columns = {}
        for k, create_data in data.items():
            comps = create_data.split("+|+")
            if k.lower() == "index":
                if comps[0] == Strings.objType.upper():
                    idx = Index(Strings.from_return_msg(comps[1]))
                else:
                    idx = Index(create_pdarray(comps[1]))
            else:
                if comps[0] == pdarray.objType.upper():
                    columns[k] = create_pdarray(comps[1])
                elif comps[0] == Strings.objType.upper():
                    columns[k] = Strings.from_return_msg(comps[1])
                elif comps[0] == IPv4.special_objType.upper():
                    columns[k] = IPv4(create_pdarray(comps[1]))
                elif comps[0] == Datetime.special_objType.upper():
                    columns[k] = Datetime(create_pdarray(comps[1]))
                elif comps[0] == Timedelta.special_objType.upper():
                    columns[k] = Timedelta(create_pdarray(comps[1]))
                elif comps[0] == Categorical_.objType.upper():
                    columns[k] = Categorical_.from_return_msg(comps[1])
                elif comps[0] == SegArray.objType.upper():
                    columns[k] = SegArray.from_return_msg(comps[1])
                elif comps[0] == BitVector.special_objType.upper():
                    columns[k] = BitVector.from_return_msg(comps[1])

        return cls(columns, idx)


def sorted(df, column=False):
    """
    Analogous to other python 'sorted(obj)' functions in that it returns
    a sorted copy of the DataFrame.

    If no sort key is specified, sort by the first key returned.

    Note: This fails on sorting ak.Strings, as does DataFrame.sort().

    Parameters
    ----------
    df : ak.dataframe.DataFrame
        The DataFrame to sort.

    column : str
        The name of the column to sort by.

    Returns
    -------
    ak.dataframe.DataFrame
        A sorted copy of the original DataFrame.
    """

    if not isinstance(df, DataFrame):
        raise TypeError("The sorted operation requires an DataFrame.")
    result = DataFrame(df.data)
    result.sort(column)
    return result


def intx(a, b):
    """Find all the rows that are in both dataframes. Columns should be in
    identical order.

    Note: does not work for columns of floating point values, but does work for
    Strings, pdarrays of int64 type, and Categorical *should* work.
    """

    if list(a.data) == list(b.data):
        a_cols = []
        b_cols = []
        for key, val in a.items():
            if key != "index":
                a_cols.append(val)
        for key, val in b.items():
            if key != "index":
                b_cols.append(val)
        return in1d(a_cols, b_cols)

    else:
        raise ValueError("Column mismatch.")


def intersect(a, b, positions=True, unique=False):
    """
    Find the intersection of two arkouda arrays.

    This function can be especially useful when `positions=True` so
    that the caller gets the indices of values present in both arrays.

    Parameters
    ----------
    a : ak.Strings or ak.pdarray
        An array of strings

    b : ak.Strings or ak.pdarray
        An array of strings

    positions : bool (default=True)
        Return tuple of boolean pdarrays that indicate positions in a and b
        where the values are in the intersection.

    unique : bool (default=False)
        If the number of distinct values in `a` (and `b`) is equal to the size of
        `a` (and `b`), there is a more efficient method to compute the intersection.

    Returns
    -------
    (ak.pdarray, ak.pdarray)
        The indices of `a` and `b` where any element occurs at least once in both
        arrays.
    """

    # To ensure compatibility with all types of arrays:
    if isinstance(a, pdarray) and isinstance(b, pdarray):
        intx = intersect1d(a, b)
        if not positions:
            return intx
        else:
            maska = in1d(a, intx)
            maskb = in1d(b, intx)
            return (maska, maskb)

    # It takes more effort to do this with ak.Strings arrays.
    elif isinstance(a, Strings) and isinstance(b, Strings):
        # Hash the two arrays first
        hash_a00, hash_a01 = a.hash()
        hash_b00, hash_b01 = b.hash()

        # a and b do not have duplicate entries, so the hashes are distinct
        if unique:
            hash0 = concatenate([hash_a00, hash_b00])
            hash1 = concatenate([hash_a01, hash_b01])

            # Group by the unique hashes
            gb = akGroupBy([hash0, hash1])
            val, cnt = gb.count()

            # Hash counts, in groupby order
            counts = gb.broadcast(cnt, permute=False)

            # Same, in original order
            tmp = counts[:]
            counts[gb.permutation] = tmp
            del tmp

            # Masks
            maska = (counts > 1)[: a.size]
            maskb = (counts > 1)[a.size :]

            # The intersection for each array of hash values
            if positions:
                return (maska, maskb)
            else:
                return a[maska]

        # a and b may have duplicate entries, so get the unique hash values
        else:
            gba = akGroupBy([hash_a00, hash_a01])
            gbb = akGroupBy([hash_b00, hash_b01])

            # Take the unique keys as the hash we'll work with
            a0, a1 = gba.unique_keys
            b0, b1 = gbb.unique_keys
            hash0 = concatenate([a0, b0])
            hash1 = concatenate([a1, b1])

            # Group by the unique hashes
            gb = akGroupBy([hash0, hash1])
            val, cnt = gb.count()

            # Hash counts, in groupby order
            counts = gb.broadcast(cnt, permute=False)

            # Restore the original order
            tmp = counts[:]
            counts[gb.permutation] = tmp
            del tmp

            # Broadcast back up one more level
            countsa = counts[: a0.size]
            countsb = counts[a0.size :]
            counts2a = gba.broadcast(countsa, permute=False)
            counts2b = gbb.broadcast(countsb, permute=False)

            # Restore the original orders
            tmp = counts2a[:]
            counts2a[gba.permutation] = tmp
            del tmp
            tmp = counts2b[:]
            counts2b[gbb.permutation] = tmp
            del tmp

            # Masks
            maska = counts2a > 1
            maskb = counts2b > 1

            # The intersection for each array of hash values
            if positions:
                return (maska, maskb)
            else:
                return a[maska]


def invert_permutation(perm):
    """
    Find the inverse of a permutation array.

    Parameters
    ----------
    perm : ak.pdarray
        The permutation array.

    Returns
    -------
    ak.pdarray
        The inverse of the permutation array.
    """

    # Test if the array is actually a permutation
    rng = perm.max() - perm.min()
    if (unique(perm).size != perm.size) and (perm.size != rng + 1):
        raise ValueError("The array is not a permutation.")
    return coargsort([perm, arange(perm.size)])


@typechecked
def _inner_join_merge(
    left: DataFrame,
    right: DataFrame,
    on: Union[str, List[str]],
    col_intersect: Union[str, List[str]],
    left_suffix: str = "_x",
    right_suffix: str = "_y",
) -> DataFrame:
    """
    Utilizes the ak.join.inner_join function to return an ak
    DataFrame object containing only rows that are in both
    the left and right Dataframes, (based on the "on" param),
    as well as their associated values.
    Parameters
    ----------
    left: DataFrame
        The Left DataFrame to be joined
    right: DataFrame
        The Right DataFrame to be joined
    on: Optional[Union[str, List[str]]] = None
        The name or list of names of the DataFrame column(s) to join on.
        If on is None, this defaults to the intersection of the columns in both DataFrames.
    left_suffix: str = "_x"
        A string indicating the suffix to add to columns from the left dataframe for overlapping
        column names in both left and right. Defaults to "_x"
    right_suffix: str = "_y"
        A string indicating the suffix to add to columns from the right dataframe for overlapping
        column names in both left and right. Defaults to "_y"
    Returns
    -------
    DataFrame
        Inner-Joined Arkouda DataFrame
    """
    left_cols, right_cols = left.columns.copy(), right.columns.copy()
    if isinstance(on, str):
        left_inds, right_inds = inner_join(left[on], right[on])
        new_dict = {on: left[on][left_inds]}
        left_cols.remove(on)
        right_cols.remove(on)
    else:
        left_inds, right_inds = inner_join([left[col] for col in on], [right[col] for col in on])
        new_dict = {col: left[col][left_inds] for col in on}
        for col in on:
            left_cols.remove(col)
            right_cols.remove(col)

    for col in left_cols:
        new_col = col + left_suffix if col in col_intersect else col
        new_dict[new_col] = left[col][left_inds]
    for col in right_cols:
        new_col = col + right_suffix if col in col_intersect else col
        new_dict[new_col] = right[col][right_inds]
    return DataFrame(new_dict)


def _right_join_merge(
    left: DataFrame,
    right: DataFrame,
    on: Union[str, List[str]],
    col_intersect: Union[str, List[str]],
    left_suffix: str = "_x",
    right_suffix: str = "_y",
) -> DataFrame:
    """
    Utilizes the ak.join.inner_join_merge function to return an
    ak DataFrame object containing all the rows in the right Dataframe,
    as well as corresponding rows in the left (based on the "on" param),
    and all of their associated values.
    Based on pandas merge functionality.
    Parameters
    ----------
    left: DataFrame
        The Left DataFrame to be joined
    right: DataFrame
        The Right DataFrame to be joined
    on: Optional[Union[str, List[str]]] = None
        The name or list of names of the DataFrame column(s) to join on.
        If on is None, this defaults to the intersection of the columns in both DataFrames.
    left_suffix: str = "_x"
        A string indicating the suffix to add to columns from the left dataframe for overlapping
        column names in both left and right. Defaults to "_x"
    right_suffix: str = "_y"
        A string indicating the suffix to add to columns from the right dataframe for overlapping
        column names in both left and right. Defaults to "_y"
    Returns
    -------
    DataFrame
        Right-Joined Arkouda DataFrame
    """
    in_left = _inner_join_merge(left, right, on, col_intersect, left_suffix, right_suffix)
    in_left_cols, left_cols = in_left.columns.copy(), left.columns.copy()
    if isinstance(on, str):
        left_at_on = left[on]
        right_at_on = right[on]
        left_cols.remove(on)
        in_left_cols.remove(on)
    else:
        left_at_on = [left[col] for col in on]
        right_at_on = [right[col] for col in on]
        for col in on:
            left_cols.remove(col)
            in_left_cols.remove(col)

    not_in_left = right[~in1d(right_at_on, left_at_on)]
    for col in not_in_left.columns:
        if col in left_cols:
            not_in_left[col + right_suffix] = not_in_left[col]
            not_in_left = not_in_left.drop(col, axis=1)

    nan_cols = list(set(in_left) - set(in_left).intersection(set(not_in_left)))
    for col in nan_cols:
        # Create a nan array for all values not in the left df
        nan_arr = full(len(not_in_left), np.nan)
        if in_left[col].dtype == int:
            in_left[col] = akcast(in_left[col], akfloat64)
        else:
            nan_arr = akcast(nan_arr, in_left[col].dtype)
        not_in_left[col] = nan_arr
    return DataFrame.append(in_left, not_in_left)


@typechecked
def merge(
    left: DataFrame,
    right: DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    how: str = "inner",
    left_suffix: str = "_x",
    right_suffix: str = "_y",
) -> DataFrame:
    """
    Utilizes the ak.join.inner_join_merge and the ak.join.right_join_merge
    functions to return a merged Arkouda DataFrame object
    containing rows from both DataFrames as specified by the merge
    condition (based on the "how" and "on" parameters).
    Based on pandas merge functionality.
    https://github.com/pandas-dev/pandas/blob/main/pandas/core/reshape/merge.py#L137
    Parameters
    ----------
    left: DataFrame
        The Left DataFrame to be joined
    right: DataFrame
        The Right DataFrame to be joined
    on: Optional[Union[str, List[str]]] = None
        The name or list of names of the DataFrame column(s) to join on.
        If on is None, this defaults to the intersection of the columns in both DataFrames.
    how: str = "inner"
        The merge condition.
        Must be one of "inner", "left", or "right"
    left_suffix: str = "_x"
        A string indicating the suffix to add to columns from the left dataframe for overlapping
        column names in both left and right. Defaults to "_x". Only used when how is "inner"
    right_suffix: str = "_y"
        A string indicating the suffix to add to columns from the right dataframe for overlapping
        column names in both left and right. Defaults to "_y". Only used when how is "inner"
    Returns
    -------
    DataFrame
        Joined Arkouda DataFrame

    Note: Multiple column joins are only supported for integer columns
    """
    col_intersect = list(set(left.columns) & set(right.columns))
    on = on if on is not None else col_intersect

    if not isinstance(on, str):
        if not all(
            isinstance(left[col], (pdarray, Strings)) and isinstance(right[col], (pdarray, Strings))
            for col in on
        ):
            raise ValueError("All columns of a multi-column merge must be pdarrays")

    if how == "inner":
        return _inner_join_merge(left, right, on, col_intersect, left_suffix, right_suffix)
    elif how == "right":
        return _right_join_merge(left, right, on, col_intersect, left_suffix, right_suffix)
    elif how == "left":
        return _right_join_merge(right, left, on, col_intersect, right_suffix, left_suffix)
    else:
        raise ValueError(f"Unexpected value of {how} for how. Must choose: 'inner', 'left', or 'right'")
