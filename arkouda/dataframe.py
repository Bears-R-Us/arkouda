from __future__ import annotations

import json
import os
import random
from collections import UserDict
from re import compile, match
from typing import Callable, Dict, List, Optional, Union, cast
from warnings import warn

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typeguard import typechecked

from arkouda import list_registry
from arkouda.categorical import Categorical
from arkouda.client import generic_msg, maxTransferBytes
from arkouda.client_dtypes import BitVector, Fields, IPv4
from arkouda.dtypes import bool as akbool
from arkouda.dtypes import float64 as akfloat64
from arkouda.dtypes import int64 as akint64
from arkouda.groupbyclass import GroupBy as akGroupBy
from arkouda.groupbyclass import unique
from arkouda.index import Index
from arkouda.numeric import cast as akcast
from arkouda.numeric import cumsum
from arkouda.numeric import isnan as akisnan
from arkouda.numeric import where
from arkouda.pdarrayclass import RegistrationError
from arkouda.pdarrayclass import attach as pd_attach
from arkouda.pdarrayclass import pdarray, unregister_pdarray_by_name
from arkouda.pdarraycreation import arange, array, create_pdarray, zeros
from arkouda.io import get_filetype, load_all, save_all
from arkouda.pdarraysetops import concatenate, in1d, intersect1d
from arkouda.row import Row
from arkouda.segarray import SegArray
from arkouda.series import Series
from arkouda.sorting import argsort, coargsort
from arkouda.strings import Strings
from arkouda.timeclass import Datetime

# This is necessary for displaying DataFrames with BitVector columns,
# because pandas _html_repr automatically truncates the number of displayed bits
pd.set_option("display.max_colwidth", 65)

__all__ = [
    "DataFrame",
    "sorted",
    "intersect",
    "invert_permutation",
    "intx",
]


def groupby_operators(cls):
    for name in [
        "all",
        "any",
        "argmax",
        "argmin",
        "max",
        "mean",
        "min",
        "nunique",
        "prod",
        "sum",
        "OR",
        "AND",
        "XOR",
    ]:
        setattr(cls, name, cls._make_aggop(name))
    return cls


class AggregateOps:
    """Base class for GroupBy and DiffAggregate containing common functions"""

    def _gbvar(self, values):
        """Calculate the variance in a groupby"""

        values = akcast(values, "float64")
        mean = self.gb.mean(values)
        mean_broad = self.gb.broadcast(mean[1])
        centered = values - mean_broad
        var = Series(self.gb.sum(centered * centered))
        n = self.gb.sum(~akisnan(centered))
        return var / (n[1] - 1)

    def _gbstd(self, values):
        """Calculates  the standard deviation in a groupby"""
        return self._gbvar(values) ** 0.5


@groupby_operators
class GroupBy(AggregateOps):
    """A DataFrame that has been grouped by a subset of columns"""

    def __init__(self, gb, df):
        self.gb = gb
        self.df = df
        for attr in ["nkeys", "size", "permutation", "unique_keys", "segments"]:
            setattr(self, attr, getattr(gb, attr))

    @classmethod
    def _make_aggop(cls, opname):
        def aggop(self, colname):
            return Series(self.gb.aggregate(self.df.data[colname], opname))

        return aggop

    def count(self):
        return Series(self.gb.count())

    def diff(self, colname):
        """Create a difference aggregate for the given column

        For each group, the differnce between successive values is calculated.
        Aggregate operations (mean,min,max,std,var) can be done on the results.

        Parameters
        ----------

        colname:  String. Name of the column to compute the difference on.

        Returns
        -------

        DiffAggregate : object containing the differences, which can be aggregated.

        """

        return DiffAggregate(self.gb, self.df.data[colname])

    def var(self, colname):
        """Calculate variance of the difference in each group"""
        return self._gbvar(self.df.data[colname])

    def std(self, colname):
        """Calculate standard deviation of the difference in each group"""
        return self._gbstd(self.df.data[colname])

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
class DiffAggregate(AggregateOps):
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

    def var(self):
        """Calculate variance of the difference in each group"""
        return self._gbvar(self.values)

    def std(self):
        """Calculate standard deviation of the difference in each group"""
        return self._gbstd(self.values)

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

    def __init__(self, initialdata=None, index=None):
        super().__init__()

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
            # convert the lists defining each column into a pdarray
            # pd.DataFrame.values is stored as rows, we need lists to be columns
            for key, val in initialdata.to_dict("list").items():
                self.data[key] = array(val)

            self.data.update()
            return

        # Some metadata about this dataframe.
        self._size = 0
        self._bytes = 0
        self._empty = True
        self.name: Optional[str] = None

        # Initial attempts to keep an order on the columns
        self._columns = []
        self._set_index(index)

        # Add data to the DataFrame if there is any
        if initialdata is not None:
            # Used to prevent uneven array length in initialization.
            sizes = set()

            # Initial data is a dictionary of arkouda arrays
            if type(initialdata) == dict:
                for key, val in initialdata.items():
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
            elif type(initialdata) == list:
                # Create string IDs for the columns
                keys = [str(x) for x in range(len(initialdata))]
                for key, col in zip(keys, initialdata):
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
        # print("key =", key)
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
            return DataFrame(initialdata=result, index=key)

        # Select rows or columns using a list
        if isinstance(key, (list, tuple)):
            result = DataFrame()
            if len(key) <= 0:
                return result
            if len({type(x) for x in key}) > 1:
                raise TypeError("Invalid selector: too many types in list.")
            if type(key[0]) == str:
                for k in key:
                    result.data[k] = UserDict.__getitem__(self, k)
                    result._columns.append(k)
                result._empty = False
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
            return DataFrame(initialdata=rtn_data, index=arange(self.size)[s])
        else:
            raise IndexError("Invalid selector: unknown error.")

    def __setitem__(self, key, value):
        self.update_size()

        # If this is the first column added, we must create an index column.
        add_index = False
        if self._empty:
            add_index = True

        # Set a single row in the dataframe using a dict of values
        if type(key) == int:
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
        elif type(key) == str:
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
        newdf._set_index(idx)
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
                df_dict[msg[1]] = SegArray.from_parts(create_pdarray(eles[0]), create_pdarray(eles[1]))
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
                df_dict[msg[1]] = Datetime(create_pdarray(msg[2]), unit=self[msg[1]].unit)
            elif t == "BitVector":
                df_dict[msg[1]] = BitVector(
                    create_pdarray(msg[2]), width=self[msg[1]].width, reverse=self[msg[1]].reverse
                )
            else:
                df_dict[msg[1]] = create_pdarray(msg[2])

        new_df = DataFrame(df_dict)
        new_df._set_index(idx)
        return new_df.to_pandas(retain_index=True)[self._columns]

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
        elif isinstance(value, pdarray):
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
                if type(oldval) != type(newval):
                    raise TypeError("Replacement value must have the same type as the original value")
                obj.index.values[obj.index.values == oldval] = newval
        elif isinstance(mapper, dict):
            for key, val in mapper.items():
                if type(key) != type(val):
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

    def GroupBy(self, keys, use_series=False):
        """
        Group the dataframe by a column or a list of columns.

        Parameters
        ----------
        keys : string or list
            An (ordered) list of column names or a single string to group by.
        use_series : If True, returns an ak.GroupBy oject. Otherwise an arkouda GroupBy object

        Returns
        -------
        GroupBy
            Either an ak GroupBy or an arkouda GroupBy object.

        See Also
        --------
        arkouda.GroupBy
        """

        self.update_size()
        if isinstance(keys, str):
            cols = self.data[keys]
        elif not isinstance(keys, (list, tuple)):
            raise TypeError("keys must be a colum name or a list/tuple of column names")
        elif len(keys) == 1:
            cols = self.data[keys[0]]
        else:
            cols = [self.data[col] for col in keys]
        gb = akGroupBy(cols)
        if use_series:
            gb = GroupBy(gb, self)
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
            data["Index"] = self.index
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

        Notes
        -----
        This method saves one file per locale of the arkouda server. All
        files are prefixed by the path argument and suffixed by their
        locale number.
        """
        from arkouda.io import to_hdf
        data = self._prep_data(index=index, columns=columns)
        to_hdf(data, prefix_path=path, file_type=file_type)

    def to_parquet(self, path, index=False, columns=None):
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
        file_type: str (single | distribute)
            Default: distribute
            Whether to save to a single file or distribute across Locales

        Notes
        -----
        This method saves one file per locale of the arkouda server. All
        files are prefixed by the path argument and suffixed by their
        locale number.
        """
        from arkouda.io import to_parquet
        data = self._prep_data(index=index, columns=columns)
        to_parquet(data, prefix_path=path)

    def save(self, path, index=False, columns=None, file_format="HDF5"):
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

        Notes
        -----
        This method saves one file per locale of the arkouda server. All
        files are prefixed by the path argument and suffixed by their
        locale number.
        """
        warn(
            "ak.DataFrame.save has been deprecated. "
            "Please use ak.DataFrame.to_hdf or ak.DataFrame.to_parquet",
            DeprecationWarning,
        )
        # if no columns are stored, we will save all columns
        if columns is None:
            data = self.data
        else:
            data = {c: self.data[c] for c in columns}

        if index:
            data["Index"] = self.index
        save_all(data, prefix_path=path, file_format=file_format)

    @classmethod
    def load(cls, prefix_path, file_format="INFER"):
        """
        Load dataframe from file
        file_format needed for consistency with other load functions
        """
        prefix, extension = os.path.splitext(prefix_path)
        first_file = f"{prefix}_LOCALE0000{extension}"
        filetype = get_filetype(first_file) if file_format.lower() == "infer" else file_format

        # columns load backwards
        df_dict = load_all(prefix_path, file_format=filetype)

        # this assumes segments will always have corresponding values.
        # This should happen due to save config
        seg_cols = [col.split("_")[0] for col in df_dict.keys() if col.endswith("_segments")]
        df_dict_keys = [
            col.split("_")[0] if col.endswith("_segments") or col.endswith("_values") else col
            for col in df_dict.keys()
        ]

        # update dict to contain segarrays where applicable if any exist
        if len(seg_cols) > 0:
            df_dict = {
                col: SegArray.from_parts(df_dict[col + "_segments"], df_dict[col + "_values"])
                if col in seg_cols
                else df_dict[col]
                for col in df_dict_keys
            }

        df = cls(df_dict)
        if filetype == "HDF5":
            return df
        else:
            # return the dataframe with them reversed so they match what was saved
            # This is only an issue with parquet
            return df[df.columns[::-1]]

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
            if isinstance(self[key], pdarray) and self[key].dtype in (akint64, akfloat64):
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

    def groupby(self, keys, use_series=True):
        """Group the dataframe by a column or a list of columns.  Alias for GroupBy

        Parameters
        ----------
        keys : a single column name or a list of column names
        use_series : Change return type to Arkouda Groupby object.

        Returns
        -------
        An arkouda Groupby instance
        """

        return self.GroupBy(keys, use_series)

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
                [array([0]), cumsum(array([self.data[col].size for col in self.columns]))]
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

        self.index.register(f"df_index_{user_defined_name}")
        array(self.columns).register(f"df_columns_{user_defined_name}")

        for col, data in self.data.items():
            data.register(f"df_data_{data.objtype}_{col}_{user_defined_name}")

        self.name = user_defined_name
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

        if not self.name:
            raise RegistrationError(
                "This DataFrame does not have a name and does not appear to be registered."
            )

        DataFrame.unregister_dataframe_by_name(self.name)

        self.name = None  # Clear our internal DataFrame object name

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

        if self.name is None:
            return False

        # Total number of registered parts should equal number of columns plus an entry
        # for the index and the column order
        total = len(self.data) + 2
        registered = sum(data.is_registered() for col, data in self.data.items())

        if self.index.values.is_registered():
            registered += 1
        if f"df_columns_{self.name}" in list_registry():
            registered += 1

        if 0 < registered < total:
            warn(
                f"WARNING: DataFrame {self.name} expected {total} components to be registered,"
                f" but only located {registered}."
            )

        return registered == total

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
        register, is_registered, unregister, unregister_groupby_by_name
        """

        col_resp = cast(
            str, generic_msg(cmd="stringsToJSON", args={"name": f"df_columns_{user_defined_name}"})
        )
        columns = dict.fromkeys(json.loads(col_resp))
        matches = []
        regEx = compile(
            f"^df_data_({pdarray.objtype}|{Strings.objtype}|"
            f"{Categorical.objtype}|{SegArray.objtype})_.*_{user_defined_name}"
        )
        # Using the regex, cycle through the registered items and find all the columns in the DataFrame
        for name in list_registry():
            x = match(regEx, name)
            if x is not None:
                matches.append(x.group())

        if len(matches) == 0:
            raise RegistrationError(f"No registered elements with name '{user_defined_name}'")

        # Remove duplicates caused by multiple components in Categorical or SegArray and
        # loop through
        for name in set(matches):
            colName = DataFrame._parse_col_name(name, user_defined_name)[0]
            if f"_{Strings.objtype}_" in name:
                columns[colName] = Strings.attach(name)
            elif f"_{pdarray.objtype}_" in name:
                columns[colName] = pd_attach(name)
            elif f"_{Categorical.objtype}_" in name:
                columns[colName] = Categorical.attach(name)
            elif f"_{SegArray.objtype}_" in name:
                columns[colName] = SegArray.attach(name)

        index_resp = cast(
            str, generic_msg(cmd="attach", args={"name": f"df_index_{user_defined_name}_key"})
        )
        dtype = index_resp.split()[2]
        if dtype == Strings.objtype:
            ind = Strings.from_return_msg(index_resp)
        else:  # pdarray
            ind = create_pdarray(index_resp)

        index = Index.factory(ind)

        df = DataFrame(columns, index)
        df.name = user_defined_name
        return df

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

        matches = []
        regEx = compile(
            f"^df_data_({pdarray.objtype}|{Strings.objtype}|"
            f"{Categorical.objtype}|{SegArray.objtype})_.*_{user_defined_name}"
        )
        # Using the regex, cycle through the registered items and find all the columns in the DataFrame
        for name in list_registry():
            x = match(regEx, name)
            if x is not None:
                matches.append(x.group())

        if len(matches) == 0:
            raise RegistrationError(f"No registered elements with name '{user_defined_name}'")

        # Remove duplicates caused by multiple components in categorical and loop through
        for name in set(matches):
            if f"_{Strings.objtype}_" in name:
                Strings.unregister_strings_by_name(name)
            elif f"_{pdarray.objtype}_" in name:
                unregister_pdarray_by_name(name)
            elif f"_{Categorical.objtype}_" in name:
                Categorical.unregister_categorical_by_name(name)
            elif f"_{SegArray.objtype}_" in name:
                SegArray.unregister_segarray_by_name(name)

        unregister_pdarray_by_name(f"df_index_{user_defined_name}_key")
        Strings.unregister_strings_by_name(f"df_columns_{user_defined_name}")

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

    @staticmethod
    def from_return_msg(repMsg):
        """
        Creates and returns a DataFrame based on return components from ak.util.attach

        Parameters
        ----------
        repMsg : string
            A '+' delimited string of the DataFrame components to parse.

        Returns
        -------
        DataFrame
            A DataFrame representing a set of DataFrame components on the server

        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown in the process of creating
            the DataFrame instance
        """
        parts = repMsg.split("+")
        dfName = parts[1]
        cols = dict.fromkeys(json.loads(parts[2][4:]))

        # index could be a pdarray or a Strings
        idxType = parts[3].split()[2]
        if idxType == Strings.objtype:
            idx = Index.factory(Strings.from_return_msg(f"{parts[3]}+{parts[4]}"))
            i = 5
        else:  # pdarray
            idx = Index.factory(create_pdarray(parts[3]))
            i = 4

        # Column parsing
        while i < len(parts):
            if parts[i][:7] == "created":
                colName, colType = DataFrame._parse_col_name(parts[i], dfName)
                if colType == "pdarray":
                    cols[colName] = create_pdarray(parts[i])
                elif colType == "str":
                    cols[colName] = Strings.from_return_msg(f"{parts[i]}+{parts[i+1]}")
                    i += 1
                else:
                    raise ValueError(f"Unknown object type defined in return message - {colType}")

            elif parts[i] == "categorical":
                colName = DataFrame._parse_col_name(parts[i + 1], dfName)[0]
                catMsg = (
                    f"{parts[i]}+{parts[i+1]}+{parts[i+2]}+{parts[i+3]}+"
                    f"{parts[i+4]}+{parts[i+5]}+{parts[i+6]}"
                )
                cols[colName] = Categorical.from_return_msg(catMsg)
                i += 6

            elif parts[i] == "segarray":
                colName = DataFrame._parse_col_name(parts[i + 1], dfName)[0]
                cols[colName] = SegArray.from_return_msg(parts[i + 2])
                i += 2

            i += 1

        df = DataFrame(cols, idx)
        df.name = dfName
        return df


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
