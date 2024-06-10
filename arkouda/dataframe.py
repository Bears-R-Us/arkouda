from __future__ import annotations

import json
import os
import random
from collections import UserDict
from typing import Callable, Dict, List, Optional, Tuple, Union, cast
from warnings import warn

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from numpy import ndarray
from numpy._typing import _8Bit, _16Bit, _32Bit, _64Bit
from typeguard import typechecked

from arkouda.alignment import find as akfind
from arkouda.categorical import Categorical
from arkouda.client import generic_msg, maxTransferBytes
from arkouda.client_dtypes import BitVector, Fields, IPv4
from arkouda.dtypes import BigInt
from arkouda.dtypes import bool as akbool
from arkouda.dtypes import dtype
from arkouda.dtypes import float64 as akfloat64
from arkouda.dtypes import int64 as akint64
from arkouda.dtypes import resolve_scalar_dtype
from arkouda.dtypes import uint64 as akuint64
from arkouda.groupbyclass import GROUPBY_REDUCTION_TYPES
from arkouda.groupbyclass import GroupBy as akGroupBy
from arkouda.groupbyclass import unique
from arkouda.index import Index, MultiIndex
from arkouda.join import inner_join
from arkouda.numeric import cast as akcast
from arkouda.numeric import cumsum, where
from arkouda.pdarrayclass import RegistrationError
from arkouda.pdarrayclass import any as akany
from arkouda.pdarrayclass import pdarray
from arkouda.pdarrayclass import sum as aksum
from arkouda.pdarraycreation import arange, array, create_pdarray, full, zeros
from arkouda.pdarraysetops import concatenate, in1d, indexof1d, intersect1d
from arkouda.row import Row
from arkouda.segarray import SegArray
from arkouda.series import Series, is_supported_scalar
from arkouda.sorting import argsort, coargsort
from arkouda.strings import Strings
from arkouda.timeclass import Datetime, Timedelta

# This is necessary for displaying DataFrames with BitVector columns,
# because pandas _html_repr automatically truncates the number of displayed bits
pd.set_option("display.max_colwidth", 65)

__all__ = [
    "DataFrame",
    "DataFrameGroupBy",
    "DiffAggregate",
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
class DataFrameGroupBy:
    """
    A DataFrame that has been grouped by a subset of columns.

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
    gb : arkouda.groupbyclass.GroupBy
        GroupBy object, where the aggregation keys are values of column(s) of a dataframe,
        usually in preparation for aggregating with respect to the other columns.
    df : arkouda.dataframe.DataFrame
        The dataframe containing the original data.
    gb_key_names : str or list(str)
        The column name(s) associated with the aggregated columns.
    as_index : bool, default=True
        If True the grouped values of the aggregation keys will be treated as an index.
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
        arkouda.dataframe.DataFrame or arkouda.series.Series

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({"A":[1,2,2,3],"B":[3,4,5,6]})
        >>> display(df)

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   1 |   3 |
        +----+-----+-----+
        |  1 |   2 |   4 |
        +----+-----+-----+
        |  2 |   2 |   5 |
        +----+-----+-----+
        |  3 |   3 |   6 |
        +----+-----+-----+

        >>> df.groupby("A").size(as_series = False)

        +----+---------+
        |    |   size  |
        +====+=========+
        |  0 |       1 |
        +----+---------+
        |  1 |       2 |
        +----+---------+
        |  2 |       1 |
        +----+---------+

        """
        if as_series is True or (as_series is None and self.as_index is True):
            return self._return_agg_series(self.gb.size(), sort_index=sort_index)
        else:
            return self._return_agg_dataframe(self.gb.size(), "size", sort_index=sort_index)

    def sample(self, n=None, frac=None, replace=False, weights=None, random_state=None):
        """
        Return a random sample from each group. You can either specify the number of elements
        or the fraction of elements to be sampled. random_state can be used for reproducibility

        Parameters
        ----------
        n: int, optional
            Number of items to return for each group.
            Cannot be used with frac and must be no larger than
            the smallest group unless replace is True.
            Default is one if frac is None.

        frac: float, optional
            Fraction of items to return. Cannot be used with n.

        replace: bool, default False
            Allow or disallow sampling of the same row more than once.

        weights: pdarray, optional
            Default None results in equal probability weighting.
            If passed a pdarray, then values must have the same length as the underlying DataFrame
            and will be used as sampling probabilities after normalization within each group.
            Weights must be non-negative with at least one positive element within each group.

        random_state: int or ak.random.Generator, optional
            If int, seed for random number generator.
            If ak.random.Generator, use as given.

        Returns
        -------
        DataFrame
            A new DataFrame containing items randomly sampled from each group
            sorted according to the grouped columns.

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({"A":[3,1,2,1,2,3],"B":[3,4,5,6,7,8]})
        >>> display(df)
        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   3 |   3 |
        +----+-----+-----+
        |  1 |   1 |   4 |
        +----+-----+-----+
        |  2 |   2 |   5 |
        +----+-----+-----+
        |  3 |   1 |   6 |
        +----+-----+-----+
        |  4 |   2 |   7 |
        +----+-----+-----+
        |  5 |   3 |   8 |
        +----+-----+-----+

        >>> df.groupby("A").sample(random_state=6)

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  3 |   1 |   6 |
        +----+-----+-----+
        |  4 |   2 |   7 |
        +----+-----+-----+
        |  5 |   3 |   8 |
        +----+-----+-----+

        >>> df.groupby("A").sample(frac=0.5, random_state=3, weights=ak.array([1,1,1,0,0,0]))

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  1 |   1 |   4 |
        +----+-----+-----+
        |  2 |   2 |   5 |
        +----+-----+-----+
        |  0 |   3 |   3 |
        +----+-----+-----+

        >>> df.groupby("A").sample(n=3, replace=True, random_state=ak.random.default_rng(7))
        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  1 |   1 |   4 |
        +----+-----+-----+
        |  3 |   1 |   6 |
        +----+-----+-----+
        |  1 |   1 |   4 |
        +----+-----+-----+
        |  4 |   2 |   7 |
        +----+-----+-----+
        |  4 |   2 |   7 |
        +----+-----+-----+
        |  4 |   2 |   7 |
        +----+-----+-----+
        |  0 |   3 |   3 |
        +----+-----+-----+
        |  5 |   3 |   8 |
        +----+-----+-----+
        |  5 |   3 |   8 |
        +----+-----+-----+
        """
        return self.df.loc[
            self.gb.sample(
                values=self.df.index.values,
                n=n,
                frac=frac,
                replace=replace,
                weights=weights,
                random_state=random_state,
                return_indices=True,
                permute_samples=True,
            )
        ]

    def _return_agg_series(self, values, sort_index=True):
        if self.as_index is True:
            if isinstance(self.gb_key_names, str):
                # handle when values is a tuple/list containing data and index
                # since we are also sending the index keyword
                if isinstance(values, (Tuple, List)) and len(values) == 2:
                    _, values = values

                series = Series(values, index=Index(self.gb.unique_keys, name=self.gb_key_names))
            elif isinstance(self.gb_key_names, list) and len(self.gb_key_names) == 1:
                # handle when values is a tuple/list containing data and index
                # since we are also sending the index keyword
                if isinstance(values, (Tuple, List)) and len(values) == 2:
                    _, values = values

                series = Series(values, index=Index(self.gb.unique_keys, name=self.gb_key_names[0]))
            elif isinstance(self.gb_key_names, list) and len(self.gb_key_names) > 1:
                from arkouda.index import MultiIndex

                # handle when values is a tuple/list containing data and index
                # since we are also sending the index keyword
                if isinstance(values, (Tuple, List)) and len(values) == 2:
                    _, values = values

                series = Series(
                    values,
                    index=MultiIndex(self.gb.unique_keys, names=self.gb_key_names),
                )
        else:
            series = Series(values)

        if sort_index is True:
            series = series.sort_index()

        return series

    def _return_agg_dataframe(self, values, name, sort_index=True):
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
        """
        Create a difference aggregate for the given column.

        For each group, the difference between successive values is calculated.
        Aggregate operations (mean,min,max,std,var) can be done on the results.

        Parameters
        ----------
        colname:  str
            Name of the column to compute the difference on.

        Returns
        -------
        DiffAggregate
            Object containing the differences, which can be aggregated.

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({"A":[1,2,2,2,3,3],"B":[3,9,11,27,86,100]})
        >>> display(df)

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   1 |   3 |
        +----+-----+-----+
        |  1 |   2 |   9 |
        +----+-----+-----+
        |  2 |   2 |  11 |
        +----+-----+-----+
        |  3 |   2 |  27 |
        +----+-----+-----+
        |  4 |   3 |  86 |
        +----+-----+-----+
        |  5 |   3 | 100 |
        +----+-----+-----+

        >>> gb = df.groupby("A")
        >>> gb.diff("B").values
        array([nan nan 2.00000000000000000 16.00000000000000000 nan 14.00000000000000000])

        """

        return DiffAggregate(self.gb, self.df.data[colname])

    def broadcast(self, x, permute=True):
        """
        Fill each group’s segment with a constant value.

        Parameters
        ----------
        x :  Series or pdarray
            The values to put in each group’s segment.
        permute : bool, default=True
            If True (default), permute broadcast values back to the
            ordering of the original array on which GroupBy was called.
            If False, the broadcast values are grouped by value.

        Returns
        -------
        arkouda.series.Series
            A Series with the Index of the original frame and the values of the broadcast.

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> from arkouda.dataframe import DataFrameGroupBy
        >>> df = ak.DataFrame({"A":[1,2,2,3],"B":[3,4,5,6]})

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   1 |   3 |
        +----+-----+-----+
        |  1 |   2 |   4 |
        +----+-----+-----+
        |  2 |   2 |   5 |
        +----+-----+-----+
        |  3 |   3 |   6 |
        +----+-----+-----+

        >>> gb = df.groupby("A")
        >>> x = ak.array([10,11,12])
        >>> s = DataFrameGroupBy.broadcast(gb, x)
        >>> df["C"] = s.values
        >>> display(df)

        +----+-----+-----+-----+
        |    |   A |   B |   C |
        +====+=====+=====+=====+
        |  0 |   1 |   3 |  10 |
        +----+-----+-----+-----+
        |  1 |   2 |   4 |  11 |
        +----+-----+-----+-----+
        |  2 |   2 |   5 |  11 |
        +----+-----+-----+-----+
        |  3 |   3 |   6 |  12 |
        +----+-----+-----+-----+

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

    Attributes
    ----------
    gb : arkouda.groupbyclass.GroupBy
        GroupBy object, where the aggregation keys are values of column(s) of a dataframe.
    values : arkouda.series.Series.
        A column to compute the difference on.
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
    >>> ak.connect()
    >>> df = ak.DataFrame()
    >>> df['a'] = ak.array([1,2,3])
    >>> display(df)

    +----+-----+
    |    |   a |
    +====+=====+
    |  0 |   1 |
    +----+-----+
    |  1 |   2 |
    +----+-----+
    |  2 |   3 |
    +----+-----+

    Create a new DataFrame using a dictionary of data:

    >>> userName = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
    >>> userID = ak.array([111, 222, 111, 333, 222, 111])
    >>> item = ak.array([0, 0, 1, 1, 2, 0])
    >>> day = ak.array([5, 5, 6, 5, 6, 6])
    >>> amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
    >>> df = ak.DataFrame({'userName': userName, 'userID': userID,
    >>>            'item': item, 'day': day, 'amount': amount})
    >>> display(df)

    +----+------------+----------+--------+-------+----------+
    |    | userName   |   userID |   item |   day |   amount |
    +====+============+==========+========+=======+==========+
    |  0 | Alice      |      111 |      0 |     5 |      0.5 |
    +----+------------+----------+--------+-------+----------+
    |  1 | Bob        |      222 |      0 |     5 |      0.6 |
    +----+------------+----------+--------+-------+----------+
    |  2 | Alice      |      111 |      1 |     6 |      1.1 |
    +----+------------+----------+--------+-------+----------+
    |  3 | Carol      |      333 |      1 |     5 |      1.2 |
    +----+------------+----------+--------+-------+----------+
    |  4 | Bob        |      222 |      2 |     6 |      4.3 |
    +----+------------+----------+--------+-------+----------+
    |  5 | Alice      |      111 |      0 |     6 |      0.6 |
    +----+------------+----------+--------+-------+----------+

    Indexing works slightly differently than with pandas:

    >>> df[0]

    +------------+----------+
    | keys       |   values |
    +============+==========+
    | userName   |    Alice |
    +------------+----------+
    |userID      |      111 |
    +------------+----------+
    | item       |      0   |
    +------------+----------+
    | day        |      5   |
    +------------+----------+
    | amount     |     0.5  |
    +------------+----------+

    >>> df['userID']
    array([111, 222, 111, 333, 222, 111])

    >>> df['userName']
    array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])

    >>> df[ak.array([1,3,5])]

    +----+------------+----------+--------+-------+----------+
    |    | userName   |   userID |   item |   day |   amount |
    +====+============+==========+========+=======+==========+
    |  0 | Bob        |      222 |      0 |     5 |      0.6 |
    +----+------------+----------+--------+-------+----------+
    |  1 | Carol      |      333 |      1 |     5 |      1.2 |
    +----+------------+----------+--------+-------+----------+
    |  2 | Alice      |      111 |      0 |     6 |      0.6 |
    +----+------------+----------+--------+-------+----------+

    Compute the stride:

    >>> df[1:5:1]

    +----+------------+----------+--------+-------+----------+
    |    | userName   |   userID |   item |   day |   amount |
    +====+============+==========+========+=======+==========+
    |  0 | Bob        |      222 |      0 |     5 |      0.6 |
    +----+------------+----------+--------+-------+----------+
    |  1 | Alice      |      111 |      1 |     6 |      1.1 |
    +----+------------+----------+--------+-------+----------+
    |  2 | Carol      |      333 |      1 |     5 |      1.2 |
    +----+------------+----------+--------+-------+----------+
    |  3 | Bob        |      222 |      2 |     6 |      4.3 |
    +----+------------+----------+--------+-------+----------+

    >>> df[ak.array([1,2,3])]

    +----+------------+----------+--------+-------+----------+
    |    | userName   |   userID |   item |   day |   amount |
    +====+============+==========+========+=======+==========+
    |  0 | Bob        |      222 |      0 |     5 |      0.6 |
    +----+------------+----------+--------+-------+----------+
    |  1 | Alice      |      111 |      1 |     6 |      1.1 |
    +----+------------+----------+--------+-------+----------+
    |  2 | Carol      |      333 |      1 |     5 |      1.2 |
    +----+------------+----------+--------+-------+----------+

    >>> df[['userID', 'day']]

    +----+----------+-------+
    |    |   userID |   day |
    +====+==========+=======+
    |  0 |      111 |     5 |
    +----+----------+-------+
    |  1 |      222 |     5 |
    +----+----------+-------+
    |  2 |      111 |     6 |
    +----+----------+-------+
    |  3 |      333 |     5 |
    +----+----------+-------+
    |  4 |      222 |     6 |
    +----+----------+-------+
    |  5 |      111 |     6 |
    +----+----------+-------+

    """

    _COLUMN_CLASSES = (pdarray, Strings, Categorical, SegArray)

    objType = "DataFrame"

    def __init__(self, initialdata=None, index=None, columns=None):
        super().__init__()
        self.registered_name = None

        if isinstance(initialdata, DataFrame):
            # Copy constructor
            self._nrows = initialdata._nrows
            self._bytes = initialdata._bytes
            self._empty = initialdata._empty
            self._columns = initialdata._columns
            if index is None:
                self._set_index(initialdata.index)
            else:
                self._set_index(index)
            self.data = initialdata.data
            self.update_nrows()
            return
        elif isinstance(initialdata, pd.DataFrame):
            # copy pd.DataFrame data into the ak.DataFrame object
            self._nrows = initialdata.shape[0]
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
        self._nrows = 0
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
                    if isinstance(val, Series):
                        val = val.values
                    if not isinstance(val, self._COLUMN_CLASSES):
                        raise ValueError(f"Values must be one of {self._COLUMN_CLASSES}.")
                    if isinstance(key, str) and key.lower() == "index":
                        # handles the index as an Index object instead of a column
                        self._set_index(val)
                        continue
                    sizes.add(val.size)
                    if len(sizes) > 1:
                        raise ValueError("Input arrays must have equal size.")
                    self._empty = False
                    self[key] = val

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
                    if not isinstance(col, self._COLUMN_CLASSES):
                        raise ValueError(f"Values must be one of {self._COLUMN_CLASSES}.")
                    sizes.add(col.size)
                    if len(sizes) > 1:
                        raise ValueError("Input arrays must have equal size.")
                    self._empty = False
                    self[key] = col

            # Initial data is invalid.
            else:
                raise ValueError(f"Initialize with dict or list of {self._COLUMN_CLASSES}.")

            # Update the dataframe indices and metadata.
            if len(sizes) > 0:
                self._nrows = sizes.pop()

            # If the index param was passed in, use that instead of
            # creating a new one.
            if index is not None:
                self._set_index(index)
            elif self.index is None:
                self._set_index(arange(self._nrows))

            self.update_nrows()

    def __getattr__(self, key):
        if key not in self.columns.values:
            raise AttributeError(f"Attribute {key} not found")
        # Should this be cached?
        return self[key]

    def __dir__(self):
        return dir(DataFrame) + self.columns.values + ["columns"]

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
        self.update_nrows()

    def validate_key(self, key):
        if key is None:
            raise ValueError("Key cannot be None")
        if isinstance(key, Series):
            # TODO: check index alignment
            return self.validate_key(key.values)
        if isinstance(key, list):
            return self.validate_key(array(key))

        if isinstance(key, slice):
            if key.start is not None and key.start < 0:
                raise ValueError("Negative start index not supported")
            if key.stop is not None and key.stop > len(self):
                raise ValueError("Slice stop index out of range")
            return key

        if is_supported_scalar(key):
            if len(self.columns) == 0:
                # Empty DataFrame, scalar key is valid
                return key
            if self.column_label_type() != resolve_scalar_dtype(key):
                raise TypeError(f"Expected key of type {self.column_label_type()}, received {type(key)}")
            return key

        if isinstance(key, pdarray) and key.dtype == akbool:
            if len(key) != len(self):
                raise ValueError("Boolean mask arguments must have the same length as the DataFrame.")
            return key

        if isinstance(key, (pdarray, Strings, Categorical, SegArray)):
            k = key[0]
            if len(self.columns) != 0 and resolve_scalar_dtype(k) != self.column_label_type():
                raise TypeError(
                    "Expected key of type {}, received {}".format(type(self.columns[0]), type(k))
                )
            return key

        raise TypeError("Indexing with keys of type {} not supported".format(type(key)))

    def column_label_type(self):
        if len(self._columns) != 0:
            return resolve_scalar_dtype(self._columns[0])
        else:
            return None

    def __getitem__(self, key):
        """
        Name-based indexing of columns, except for integer slices and boolean masks,
        which does position-based indexing of rows.

        Parameters
        ----------
        key : str, int, float, list, pdarray, slice
            The column label(s) the resulting Series or DataFrame should contain. If
            using a slice, the indices of the desired rows. If using a boolean mask,
            a pdarray where "True" entries correspond to desired rows.

        Returns
        -------
        Series, DataFrame
            The columns or rows of the DataFrame. If only one column label is
            provided, the return type is a Series. Otherwise a DataFrame is returned.

        Raises
        ------
        KeyError
            Raised if a column label is not present in the DataFrame
        ValueError
            Raised if a boolean mask has the incorrect length or a slice's bounds are
            out of range
        TypeError
            Raised if the key is not a supported type
        """
        # convert series to underlying values
        # Should check for index alignment
        key = self.validate_key(key)

        # if a scalar argument, return a Series
        if is_supported_scalar(key):
            if key not in self._columns:
                raise KeyError("column {} not present in DataFrame".format(key))
            values = UserDict.__getitem__(self, key)
            index = self.index
            return Series(values, index=index)

        # boolean mask
        if isinstance(key, pdarray) and key.dtype == akbool:
            return self._get_rows(key)

        # slice
        if isinstance(key, slice):
            return self._get_rows(key)

        if isinstance(key, (pdarray, Strings)):
            for k in key.to_ndarray():
                if k not in self.columns:
                    raise KeyError("column {} not present in DataFrame".format(k))
            result = DataFrame()
            if len(key) <= 0:
                return result
            for k in key.to_ndarray():
                result.data[k] = UserDict.__getitem__(self, k)
                result._columns.append(k)
            result._empty = False
            result._set_index(self.index)
            return result

        raise TypeError("key not supported: {}".format(key))

    def validate_value(self, value):
        if isinstance(value, Series):
            # TODO: check index alignment
            return self.validate_value(value.values)
        if isinstance(value, list):
            return self.validate_value(array(value))
        if isinstance(value, tuple):
            raise TypeError("DataFrame does not support tuple values")

        return value

    def __setitem__(self, key, value):
        """
        Inserts/updates columns in the DataFrame. Can also be used to
        update one DataFrame with values from another.

        Parameters
        ----------
        key : str, int, float, list, pdarray
            The column label(s) the resulting Series or DataFrame should contain.
        value : str, int, float, list, pdarray, Series, DataFrame
            The value(s) that should be inserted or updated within the DataFrame.

        Raises
        ------
        KeyError
            Raised if a column label is not present in the DataFrame
        IndexError
            Raised if a boolean mask has the incorrect length
        TypeError
            Raised if the key or value are not a supported type
        """
        self.update_nrows()

        key = self.validate_key(key)
        value = self.validate_value(value)

        # adding first column
        if len(self._columns) == 0 and is_supported_scalar(key):
            self._columns.append(key)
            self._empty = False
            if is_supported_scalar(value):
                value = full(1, value, resolve_scalar_dtype(value))
            UserDict.__setitem__(self, key, value)
            self._set_index(Index(arange(len(value))))
            self.update_nrows()
            return

        # Update or insert a single column into the dataframe
        if resolve_scalar_dtype(key) == self.column_label_type():
            if is_supported_scalar(value):
                value = full(len(self), value, resolve_scalar_dtype(value))
            assert isinstance(value, (pdarray, Strings, Categorical, SegArray))
            if len(value) != len(self):
                raise ValueError("Column length must match DataFrame length")

            # Set a single column in the dataframe using a scalar value
            if key not in self.columns:
                self._empty = False
                self._columns.append(key)

            UserDict.__setitem__(self, key, value)
            return

        # Boolean mask
        elif isinstance(key, pdarray) and key.dtype == akbool:
            if not isinstance(value, DataFrame):
                raise ValueError("Expected DataFrame type for boolean mask assignment")
            shared_columns = intersect1d(array(self.columns.values), array(value.columns.values))
            if len(shared_columns) != len(self.columns) or len(shared_columns) != len(value.columns):
                raise ValueError("Right-hand side columns do not match left-hand side columns")
            if len(value) != aksum(key):
                raise IndexError("Boolean mask length must match DataFrame length")
            for k in self.columns:
                self[k].values[key] = value[k].values
            return

        # Index lists
        if isinstance(key, (pdarray, Strings)):
            if isinstance(value, DataFrame):
                if not len(key) == len(value.columns):
                    raise ValueError(
                        f"Number of keys and values must match: {len(key)} != {len(value.columns)}"
                    )
            else:
                raise ValueError("When setting multiple columns, value must be a DataFrame")

            for k, valueColumn in zip(key.to_ndarray(), value.columns):
                v = value[valueColumn].values
                if len(v) != len(self):
                    raise ValueError("Column length must match DataFrame length")
                if k not in self.columns:
                    self._empty = False
                    self._columns.append(k)
                UserDict.__setitem__(self, k, v)
            return

        raise TypeError("Setting on dataframe with unexpected type: {}".format(type(key)))

    @property
    def loc(self):
        """
        Label-based row indexing. Supports getting and setting. If there is
        a single indexing argument, it is interpreted as a row selector. If
        there are two, the first is interpreted as a row selector and the
        second as a column selector. Selectors can be scalar label values;
        lists, pdarrays, or slices of label values; boolean masks as pdarrays;
        or Series. Setting values with .loc requires both the row and column
        selectors to be present.

        Raises
        ------
        KeyError
            Raised if a label is not present in the DataFrame.
        TypeError
            Raised if the key or value types are not supported.
        """
        return _LocIndexer(self)

    @property
    def iloc(self):
        """
        Position-based row indexing. Supports getting and setting. If there
        is a single indexing argument, it is interpreted as a row selector.
        If there are two, the first is interpreted as a row selector and the
        second as a column selector. Selectors can be scalar integer values;
        lists, pdarrays, or slices of integer values; boolean masks as pdarrays;
        or integer Series. Setting values with .iloc requires both the row and
        column selectors to be present.

        Raises
        ------
        TypeError
            Raised if the keys or value types are not supported.

        IndexError
            Raised if an index is out of range.

        ValueError
            Raised if a boolean mask is of the wrong length.
        """
        return _ILocIndexer(self)

    @property
    def at(self) -> AtIndexer:
        """
        Access a single value for a row/column by label. Similar to `.loc`.
        Use only if you need to get or set a single value.

        Raises
        ------
        KeyError
            Raised if a label is not present in the DataFrame.
        TypeError
            Raised if the key or value types are not supported.
        """
        return AtIndexer(self)

    @property
    def iat(self) -> IAtIndexer:
        """
        Access a single value for a row/column pair by integer position.
        Similar to `.iloc`. Use only if you need to get or set a single value.

        Raises
        ------
        IndexError
            Raised if an index is out of range.
        TypeError
            Raised if the key or value types are not supported.
        """
        return IAtIndexer(self)

    def set_row(self, key, value):
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
            elif key >= self._nrows:
                raise KeyError("The row index is out of range.")
            else:
                for k, v in value.items():
                    # maintaining to prevent adding index column
                    if k == "index":
                        continue
                    self[k][key] = v

    def __len__(self):
        """
        Return the number of rows.
        """
        return self._nrows

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

        self.update_nrows()

        if self._empty:
            return "DataFrame([ -- ][ 0 rows : 0 B])"

        keys = [str(key) for key in list(self._columns)]
        keys = [("'" + key + "'") for key in keys]
        keystr = ", ".join(keys)

        # first call to memory_usage_info() initializes self._bytes
        mem = self.memory_usage_info()

        # Get units that make the most sense.
        if self._bytes < 1024:
            mem = self.memory_usage_info(unit="B")
        elif self._bytes < 1024**2:
            mem = self.memory_usage_info(unit="KB")
        elif self._bytes < 1024**3:
            mem = self.memory_usage_info(unit="MB")
        else:
            mem = self.memory_usage_info(unit="GB")
        rows = " rows"
        if self._nrows == 1:
            rows = " row"
        return "DataFrame([" + keystr + "], {:,}".format(self._nrows) + rows + ", " + str(mem) + ")"

    def _get_head_tail(self):
        if self._empty:
            return pd.DataFrame()
        self.update_nrows()
        maxrows = pd.get_option("display.max_rows")
        if self._nrows <= maxrows:
            newdf = DataFrame()
            for col in self._columns:
                if isinstance(self[col], Categorical):
                    newdf[col] = self[col].categories[self[col].codes]
                else:
                    newdf[col] = self[col]
            newdf._set_index(self.index)
            return newdf.to_pandas(retain_index=True)
        # Being 1 above the threshold causes the PANDAS formatter to split the data frame vertically
        idx = array(
            list(range(maxrows // 2 + 1)) + list(range(self._nrows - (maxrows // 2), self._nrows))
        )
        newdf = DataFrame()
        for col in self._columns:
            if isinstance(self[col], Categorical):
                newdf[col] = self[col].categories[self[col].codes[idx]]
            else:
                newdf[col] = self[col].iloc[idx]
        newdf._set_index(self.index.index[idx])
        return newdf.to_pandas(retain_index=True)

    def _get_head_tail_server(self):
        if self._empty:
            return pd.DataFrame()
        self.update_nrows()
        maxrows = pd.get_option("display.max_rows")
        if self._nrows <= maxrows:
            newdf = DataFrame()
            for col in self._columns:
                if isinstance(self[col].values, Categorical):
                    newdf[col] = self[col].values.categories[self[col].values.codes]
                else:
                    newdf[col] = self[col].values
            newdf._set_index(self.index)
            return newdf.to_pandas(retain_index=True)
        # Being 1 above the threshold causes the PANDAS formatter to split the data frame vertically
        idx = array(
            list(range(maxrows // 2 + 1)) + list(range(self._nrows - (maxrows // 2), self._nrows))
        )
        msg_list = []
        for col in self._columns:
            if isinstance(self[col].values, Categorical):
                msg_list.append(
                    f"Categorical+{col}+{self[col].values.codes.name}"
                    f"+{self[col].values.categories.name}"
                )
            elif isinstance(self[col].values, SegArray):
                msg_list.append(
                    f"SegArray+{col}+{self[col].values.segments.name}+{self[col].values.values.name}"
                )
            elif isinstance(self[col].values, Strings):
                msg_list.append(f"Strings+{col}+{self[col].values.name}")
            elif isinstance(self[col].values, Fields):
                msg_list.append(f"Fields+{col}+{self[col].values.name}")
            elif isinstance(self[col].values, IPv4):
                msg_list.append(f"IPv4+{col}+{self[col].values.name}")
            elif isinstance(self[col].values, Datetime):
                msg_list.append(f"Datetime+{col}+{self[col].values.name}")
            elif isinstance(self[col].values, BitVector):
                msg_list.append(f"BitVector+{col}+{self[col].values.name}")
            else:
                msg_list.append(f"pdarray+{col}+{self[col].values.name}")

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
                    self[msg[1]].values.names,
                    MSB_left=self[msg[1]].values.MSB_left,
                    pad=self[msg[1]].values.padchar,
                    separator=self[msg[1]].values.separator,
                    show_int=self[msg[1]].values.show_int,
                )
            elif t == "IPv4":
                df_dict[msg[1]] = IPv4(create_pdarray(msg[2]))
            elif t == "Datetime":
                df_dict[msg[1]] = Datetime(create_pdarray(msg[2]))
            elif t == "BitVector":
                df_dict[msg[1]] = BitVector(
                    create_pdarray(msg[2]),
                    width=self[msg[1]].values.width,
                    reverse=self[msg[1]].values.reverse,
                )
            else:
                df_dict[msg[1]] = create_pdarray(msg[2])

        new_df = DataFrame(df_dict)
        new_df._set_index(self.index.index[idx])
        return new_df.to_pandas(retain_index=True)[self._columns]

    def _get_rows(self, key):
        """
        Gets rows of the dataframe based with the provided indices
        """
        if not isinstance(key, (pdarray, slice)):
            raise TypeError("_get_rows requires pdarray of row indices or a slice")

        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else len(self)
            step = key.step if key.step is not None else 1
            key = arange(start, stop, step)
        if key.dtype == akbool:
            key = arange(key.size)[key]
        result = {}
        for k in self._columns:
            result[k] = UserDict.__getitem__(self, k)[key]
        # To stay consistent with numpy, provide the old index values
        return DataFrame(initialdata=result, index=self.index.index[key])

    def column_labels(self):
        """
        Return the column labels.
        """
        return self._columns

    def _add_new_rows(self, key):
        # If the key is a scalar, convert it to an array
        if is_supported_scalar(key) and dtype(type(key)) == self.index.dtype:
            key = array([key])

        # Cannot add new rows to dataframes with String columns
        for k in self._columns:
            if isinstance(UserDict.__getitem__(self, k), Strings):
                raise ValueError(
                    "This DataFrame has a column of type ak.Strings;"
                    " so this DataFrame is immutable. This feature could change"
                    " if arkouda supports mutable Strings in the future."
                )

        if isinstance(key, pdarray) and key.dtype == self.index.dtype:
            new_keys = key[in1d(key, self.index.values, invert=True)]
            self._set_index(self.index.concat(Index(new_keys)))
            for k in self._columns:
                current_col = UserDict.__getitem__(self, k)
                default_val = np.nan if current_col.dtype == akfloat64 else 0
                new_col = concatenate(
                    [current_col, full(len(new_keys), default_val, dtype=current_col.dtype)]
                )
                UserDict.__setitem__(self, k, new_col)

            self.update_nrows()
        else:
            raise ValueError("Invalid key type for adding new rows")

    def _add_column(self, key, dtype):
        """
        Adds a column to the DataFrame with the given key and dtype.
        """
        if key in self.columns:
            raise ValueError(f"Column {key} already exists in DataFrame")
        default_value = 0
        if dtype == akfloat64:
            default_value = np.nan
        self[key] = full(len(self), default_value, dtype=dtype)

    def transfer(self, hostname, port):
        """
        Sends a DataFrame to a different Arkouda server.

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
        str
            A message indicating a complete transfer.

        Raises
        ------
        ValueError
            Raised if the op is not within the pdarray.BinOps set
        TypeError
            Raised if other is not a pdarray or the pdarray.dtype is not
            a supported dtype
        """
        self.update_nrows()
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
        return f"{self._nrows} rows x {self._ncols()} columns"

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
        """
        Copy the data from a pandas DataFrame into a new arkouda.dataframe.DataFrame.

        Parameters
        ----------
        pd_df : pandas.DataFrame
            A pandas DataFrame to convert.

        Returns
        -------
        arkouda.dataframe.DataFrame

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> import pandas as pd
        >>> pd_df = pd.DataFrame({"A":[1,2],"B":[3,4]})
        >>> type(pd_df)
        pandas.core.frame.DataFrame
        >>> display(pd_df)

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   1 |   3 |
        +----+-----+-----+
        |  1 |   2 |   4 |
        +----+-----+-----+

        >>> ak_df = DataFrame.from_pandas(pd_df)
        >>> type(ak_df)
        arkouda.dataframe.DataFrame
        >>> display(ak_df)

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   1 |   3 |
        +----+-----+-----+
        |  1 |   2 |   4 |
        +----+-----+-----+

        """
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
            The labels to be dropped on the given axis.
        axis : int or str
            The axis on which to drop from. 0/'index' - drop rows, 1/'columns' - drop columns.
        inplace: bool, default=False
            When True, perform the operation on the calling object.
            When False, return a new object.

        Returns
        -------
        arkouda.dataframe.DataFrame or None
            DateFrame when `inplace=False`;
            None when `inplace=True`

        Examples
        ----------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> display(df)

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      1 |      3 |
        +----+--------+--------+
        |  1 |      2 |      4 |
        +----+--------+--------+

        Drop column

        >>> df.drop('col1', axis = 1)

        +----+--------+
        |    |   col2 |
        +====+========+
        |  0 |      3 |
        +----+--------+
        |  1 |      4 |
        +----+--------+

        Drop row

        >>> df.drop(0, axis = 0)

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      2 |      4 |
        +----+--------+--------+

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

        obj.update_nrows()

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
        subset : Iterable
            Iterable of column names to use to dedupe.
        keep : {'first', 'last'}, default='first'
            Determines which duplicates (if any) to keep.

        Returns
        -------
        arkouda.dataframe.DataFrame
            DataFrame with duplicates removed.

        Example
        -------

        >>> df = ak.DataFrame({'col1': [1, 2, 2, 3], 'col2': [4, 5, 5, 6]})
        >>> display(df)

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      1 |      4 |
        +----+--------+--------+
        |  1 |      2 |      5 |
        +----+--------+--------+
        |  2 |      2 |      5 |
        +----+--------+--------+
        |  3 |      3 |      6 |
        +----+--------+--------+

        >>> df.drop_duplicates()

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      1 |      4 |
        +----+--------+--------+
        |  1 |      2 |      5 |
        +----+--------+--------+
        |  2 |      3 |      6 |
        +----+--------+--------+

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
            return self.iloc[gp.permutation[_segment_ends]]
        else:
            return self.iloc[gp.permutation[gp.segments]]

    @property
    def size(self):
        """
        Returns the number of bytes on the arkouda server.

        Returns
        -------
        int
            The number of bytes on the arkouda server.

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        >>> df

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      1 |      4 |
        +----+--------+--------+
        |  1 |      2 |      5 |
        +----+--------+--------+
        |  2 |      3 |      6 |
        +----+--------+--------+

        >>> df.size
        6
        """

        self.update_nrows()
        if self._nrows is None:
            return 0
        return self.shape[0] * self.shape[1]

    @property
    def dtypes(self):
        """
        The dtypes of the dataframe.

        Returns
        -------
        dtypes :  arkouda.row.Row
            The dtypes of the dataframe.

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({'col1': [1, 2], 'col2': ["a", "b"]})
        >>> df

        +----+--------+--------+
        |    |   col1 | col2   |
        +====+========+========+
        |  0 |      1 | a      |
        +----+--------+--------+
        |  1 |      2 | b      |
        +----+--------+--------+

        >>> df.dtypes

        +----+--------+
        |keys| values |
        +====+========+
        |col1|  int64 |
        +----+--------+
        |col2|    str |
        +----+--------+

        """
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
        """
        Whether the dataframe is empty.

        Returns
        -------
        bool
            True if the dataframe is empty, otherwise False.

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({})
        >>> df
         0 rows x 0 columns
        >>> df.empty
        True
        """
        return self._empty

    @property
    def shape(self):
        """
        The shape of the dataframe.

        Returns
        -------
        tuple of int
            Tuple of array dimensions.

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        >>> df

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      1 |      4 |
        +----+--------+--------+
        |  1 |      2 |      5 |
        +----+--------+--------+
        |  2 |      3 |      6 |
        +----+--------+--------+

        >>> df.shape
        (3, 2)
        """
        self.update_nrows()
        num_cols = len(self._columns)
        nrows = self._nrows
        return (nrows, num_cols)

    @property
    def columns(self):
        """
        An Index where the values are the column names of the dataframe.

        Returns
        -------
        arkouda.index.Index
            The values of the index are the column names of the dataframe.

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      1 |      3 |
        +----+--------+--------+
        |  1 |      2 |      4 |
        +----+--------+--------+

        >>> df.columns
        Index(array(['col1', 'col2']), dtype='<U0')
        """
        if isinstance(self._columns, ndarray):
            column_names = self._columns.tolist()
        else:
            column_names = self._columns

        return Index(column_names, allow_list=True)

    @property
    def index(self):
        """
        The index of the dataframe.

        Returns
        -------
        arkouda.index.Index or arkouda.index.MultiIndex
            The index of the dataframe.

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      1 |      3 |
        +----+--------+--------+
        |  1 |      2 |      4 |
        +----+--------+--------+

        >>> df.index
        Index(array([0 1]), dtype='int64')
        """
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
    def reset_index(self, size: Optional[int] = None, inplace: bool = False) -> Union[None, DataFrame]:
        """
        Set the index to an integer range.

        Useful if this dataframe is the result of a slice operation from
        another dataframe, or if you have permuted the rows and no longer need
        to keep that ordering on the rows.

        Parameters
        ----------
        size : int, optional
            If size is passed, do not attempt to determine size based on
            existing column sizes. Assume caller handles consistency correctly.
        inplace: bool, default=False
            When True, perform the operation on the calling object.
            When False, return a new object.

        Returns
        -------
        arkouda.dataframe.DataFrame or None
            DateFrame when `inplace=False`;
            None when `inplace=True`.

        NOTE
        ----------
        Pandas adds a column 'index' to indicate the original index. Arkouda does not currently
        support this behavior.

        Example
        -------

        >>> df = ak.DataFrame({"A": ak.array([1, 2, 3]), "B": ak.array([4, 5, 6])})
        >>> display(df)

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   1 |   4 |
        +----+-----+-----+
        |  1 |   2 |   5 |
        +----+-----+-----+
        |  2 |   3 |   6 |
        +----+-----+-----+

        >>> perm_df = df[ak.array([0,2,1])]
        >>> display(perm_df)

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   1 |   4 |
        +----+-----+-----+
        |  1 |   3 |   6 |
        +----+-----+-----+
        |  2 |   2 |   5 |
        +----+-----+-----+

        >>> perm_df.reset_index()

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   1 |   4 |
        +----+-----+-----+
        |  1 |   3 |   6 |
        +----+-----+-----+
        |  2 |   2 |   5 |
        +----+-----+-----+

        """

        obj = self if inplace else self.copy()
        if not size:
            obj.update_nrows()
            obj._set_index(arange(obj._nrows))
        else:
            obj._set_index(arange(size))

        if not inplace:
            return obj
        return None

    @property
    def info(self):
        """
        Returns a summary string of this dataframe.

        Returns
        -------
        str
            A summary string of this dataframe.

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({'col1': [1, 2], 'col2': ["a", "b"]})
        >>> df

        +----+--------+--------+
        |    |   col1 | col2   |
        +====+========+========+
        |  0 |      1 | a      |
        +----+--------+--------+
        |  1 |      2 | b      |
        +----+--------+--------+

        >>> df.info
        "DataFrame(['col1', 'col2'], 2 rows, 20 B)"

        """

        self.update_nrows()

        if self._nrows is None:
            return "DataFrame([ -- ][ 0 rows : 0 B])"

        keys = [str(key) for key in list(self._columns)]
        keys = [("'" + key + "'") for key in keys]
        keystr = ", ".join(keys)

        # first call to memory_usage_info() initializes self._bytes
        mem = self.memory_usage_info()

        # Get units that make the most sense.
        if self._bytes < 1024:
            mem = self.memory_usage_info(unit="B")
        elif self._bytes < 1024**2:
            mem = self.memory_usage_info(unit="KB")
        elif self._bytes < 1024**3:
            mem = self.memory_usage_info(unit="MB")
        else:
            mem = self.memory_usage_info(unit="GB")
        rows = " rows"
        if self._nrows == 1:
            rows = " row"
        return "DataFrame([" + keystr + "], {:,}".format(self._nrows) + rows + ", " + str(mem) + ")"

    def items(self):
        """
        Iterate over (column name, column) pairs.

        Returns
        -------
        generator
            A generator of (column name, column) pairs.

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> for key, value in df.items():
        ...     print(key, value)
        col1 [1 2]
        col2 [3 4]
        """
        for key in self._columns:
            elt = UserDict.__getitem__(self, key)
            if isinstance(elt, Series):
                elt = elt.values
            yield key, elt

    def values(self):
        for key, elts in self.items():
            yield elts

    def update_nrows(self):
        """
        Computes the number of rows on the arkouda server and updates the size parameter.
        """
        sizes = set()
        for key, val in self.items():
            if val is not None:
                sizes.add(val.size)
        if len(sizes) > 1:
            raise ValueError("Size mismatch in DataFrame columns: ", sizes, ".")
        if len(sizes) == 0:
            self._nrows = None
        else:
            self._nrows = sizes.pop()

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
        inplace: bool, default=False
            When True, perform the operation on the calling object.
            When False, return a new object.
        Returns
        -------
        arkouda.dataframe.DataFrame or None
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
        inplace: bool, default=False
            When True, perform the operation on the calling object.
            When False, return a new object.
        Returns
        -------
        arkouda.dataframe.DataFrame or None
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
            When this is set, axis is ignored.
        axis: int or str, default=0
            Indicates which axis to perform the rename.
            0/"index" - Indexes
            1/"column" - Columns
        inplace: bool, default=False
            When True, perform the operation on the calling object.
            When False, return a new object.
        Returns
        -------
        arkouda.dataframe.DataFrame or None
            DateFrame when `inplace=False`;
            None when `inplace=True`.
        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({"A": ak.array([1, 2, 3]), "B": ak.array([4, 5, 6])})
        >>> display(df)

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   1 |   4 |
        +----+-----+-----+
        |  1 |   2 |   5 |
        +----+-----+-----+
        |  2 |   3 |   6 |
        +----+-----+-----+

        Rename columns using a mapping:

        >>> df.rename(column={'A':'a', 'B':'c'})

        +----+-----+-----+
        |    |   a |   c |
        +====+=====+=====+
        |  0 |   1 |   4 |
        +----+-----+-----+
        |  1 |   2 |   5 |
        +----+-----+-----+
        |  2 |   3 |   6 |
        +----+-----+-----+

        Rename indexes using a mapping:

        >>> df.rename(index={0:99, 2:11})

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   1 |   4 |
        +----+-----+-----+
        |  1 |   2 |   5 |
        +----+-----+-----+
        |  2 |   3 |   6 |
        +----+-----+-----+

        Rename using an axis style parameter:

        >>> df.rename(str.lower, axis='column')

        +----+-----+-----+
        |    |   a |   b |
        +====+=====+=====+
        |  0 |   1 |   4 |
        +----+-----+-----+
        |  1 |   2 |   5 |
        +----+-----+-----+
        |  2 |   3 |   6 |
        +----+-----+-----+

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
        ordered: bool, default=True
            If False, allow rows to be interleaved for better performance (but
            data within a row remains together). By default, append all rows
            to the end, in input order.

        Returns
        -------
        self
            Appending occurs in-place, but result is returned for compatibility.

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df1 = ak.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      1 |      3 |
        +----+--------+--------+
        |  1 |      2 |      4 |
        +----+--------+--------+

        >>> df2 = ak.DataFrame({'col1': [3], 'col2': [5]})

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      3 |      5 |
        +----+--------+--------+

        >>> df1.append(df2)
        >>> df1

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      1 |      3 |
        +----+--------+--------+
        |  1 |      2 |      4 |
        +----+--------+--------+
        |  2 |      3 |      5 |
        +----+--------+--------+

        """
        from arkouda.util import generic_concat as util_concatenate

        # Do nothing if the other dataframe is empty
        if other.empty:
            return self

        # Check all the columns to make sure they can be concatenated
        self.update_nrows()

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
                    tmp_data[key] = util_concatenate([self[key], other[key]], ordered=ordered).values
                except TypeError as e:
                    raise TypeError(
                        f"Incompatible types for column {key}: {type(self[key])} vs {type(other[key])}"
                    ) from e
            self.data = tmp_data
            self._set_index(self.index.concat(other.index))
        # Clean up
        self.update_nrows()
        self.reset_index(inplace=True)

        self._empty = False
        return self

    @classmethod
    def concat(cls, items, ordered=True):
        """
        Essentially an append, but different formatting.

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
        n : int, default = 5
            Number of rows to select.

        Returns
        -------
        arkouda.dataframe.DataFrame
            The first `n` rows of the DataFrame.

        See Also
        --------
        tail

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({'col1': ak.arange(10), 'col2': -1 * ak.arange(10)})
        >>> display(df)

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      0 |      0 |
        +----+--------+--------+
        |  1 |      1 |     -1 |
        +----+--------+--------+
        |  2 |      2 |     -2 |
        +----+--------+--------+
        |  3 |      3 |     -3 |
        +----+--------+--------+
        |  4 |      4 |     -4 |
        +----+--------+--------+
        |  5 |      5 |     -5 |
        +----+--------+--------+
        |  6 |      6 |     -6 |
        +----+--------+--------+
        |  7 |      7 |     -7 |
        +----+--------+--------+
        |  8 |      8 |     -8 |
        +----+--------+--------+
        |  9 |      9 |     -9 |
        +----+--------+--------+

        >>> df.head()

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      0 |      0 |
        +----+--------+--------+
        |  1 |      1 |     -1 |
        +----+--------+--------+
        |  2 |      2 |     -2 |
        +----+--------+--------+
        |  3 |      3 |     -3 |
        +----+--------+--------+
        |  4 |      4 |     -4 |
        +----+--------+--------+

        >>> df.head(n=2)

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      0 |      0 |
        +----+--------+--------+
        |  1 |      1 |     -1 |
        +----+--------+--------+

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
        n : int, default=5
            Number of rows to select.

        Returns
        -------
        arkouda.dataframe.DataFrame
            The last `n` rows of the DataFrame.

        See Also
        --------
        arkouda.dataframe.head

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({'col1': ak.arange(10), 'col2': -1 * ak.arange(10)})
        >>> display(df)

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      0 |      0 |
        +----+--------+--------+
        |  1 |      1 |     -1 |
        +----+--------+--------+
        |  2 |      2 |     -2 |
        +----+--------+--------+
        |  3 |      3 |     -3 |
        +----+--------+--------+
        |  4 |      4 |     -4 |
        +----+--------+--------+
        |  5 |      5 |     -5 |
        +----+--------+--------+
        |  6 |      6 |     -6 |
        +----+--------+--------+
        |  7 |      7 |     -7 |
        +----+--------+--------+
        |  8 |      8 |     -8 |
        +----+--------+--------+
        |  9 |      9 |     -9 |
        +----+--------+--------+

        >>> df.tail()

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      5 |     -5 |
        +----+--------+--------+
        |  1 |      6 |     -6 |
        +----+--------+--------+
        |  2 |      7 |     -7 |
        +----+--------+--------+
        |  3 |      8 |     -8 |
        +----+--------+--------+
        |  4 |      9 |     -9 |
        +----+--------+--------+

        >>> df.tail(n=2)

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      8 |     -8 |
        +----+--------+--------+
        |  1 |      9 |     -9 |
        +----+--------+--------+

        """
        self.update_nrows()
        if self._nrows <= n:
            return self
        return self[self._nrows - n :]

    def sample(self, n=5):
        """
        Return a random sample of `n` rows.

        Parameters
        ----------
        n : int, default=5
            Number of rows to return.

        Returns
        -------
        arkouda.dataframe.DataFrame
            The sampled `n` rows of the DataFrame.

        Example
        -------

        >>> df = ak.DataFrame({"A": ak.arange(5), "B": -1 * ak.arange(5)})
        >>> display(df)

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   0 |   0 |
        +----+-----+-----+
        |  1 |   1 |  -1 |
        +----+-----+-----+
        |  2 |   2 |  -2 |
        +----+-----+-----+
        |  3 |   3 |  -3 |
        +----+-----+-----+
        |  4 |   4 |  -4 |
        +----+-----+-----+

        Random output of size 3:

        >>> df.sample(n=3)

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   0 |   0 |
        +----+-----+-----+
        |  1 |   1 |  -1 |
        +----+-----+-----+
        |  2 |   4 |  -4 |
        +----+-----+-----+

        """
        self.update_nrows()
        if self._nrows <= n:
            return self
        return self[array(random.sample(range(self._nrows), n))]

    def GroupBy(self, keys, use_series=False, as_index=True, dropna=True):
        """
        Group the dataframe by a column or a list of columns.

        Parameters
        ----------
        keys : str or list of str
            An (ordered) list of column names or a single string to group by.
        use_series : bool, default=False
            If True, returns an arkouda.dataframe.DataFrameGroupBy object.
            Otherwise an arkouda.groupbyclass.GroupBy object.
        as_index: bool, default=True
            If True, groupby columns will be set as index
            otherwise, the groupby columns will be treated as DataFrame columns.
        dropna : bool, default=True
            If True, and the groupby keys contain NaN values,
            the NaN values together with the corresponding row will be dropped.
            Otherwise, the rows corresponding to NaN values will be kept.
        Returns
        -------
        arkouda.dataframe.DataFrameGroupBy or arkouda.groupbyclass.GroupBy
            If use_series = True, returns an arkouda.dataframe.DataFrameGroupBy object.
            Otherwise returns an arkouda.groupbyclass.GroupBy object.

        See Also
        --------
        arkouda.GroupBy

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({'col1': [1.0, 1.0, 2.0, np.nan], 'col2': [4, 5, 6, 7]})
        >>> df

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      1 |      4 |
        +----+--------+--------+
        |  1 |      1 |      5 |
        +----+--------+--------+
        |  2 |      2 |      6 |
        +----+--------+--------+
        |  3 |    nan |      7 |
        +----+--------+--------+

        >>> df.GroupBy("col1")
        <arkouda.groupbyclass.GroupBy at 0x7f2cf23e10c0>
        >>> df.GroupBy("col1").size()
        (array([1.00000000000000000 2.00000000000000000]), array([2 1]))

        >>> df.GroupBy("col1",use_series=True)
        col1
        1.0    2
        2.0    1
        dtype: int64
        >>> df.GroupBy("col1",use_series=True, as_index = False).size()

        +----+--------+--------+
        |    |   col1 |   size |
        +====+========+========+
        |  0 |      1 |      2 |
        +----+--------+--------+
        |  1 |      2 |      1 |
        +----+--------+--------+

        """

        self.update_nrows()
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
            gb = DataFrameGroupBy(gb, self, gb_key_names=keys, as_index=as_index)
        return gb

    def memory_usage(self, index=True, unit="B") -> Series:
        """
        Return the memory usage of each column in bytes.

        The memory usage can optionally include the contribution of
        the index.

        Parameters
        ----------
        index : bool, default True
            Specifies whether to include the memory usage of the DataFrame's
            index in returned Series. If ``index=True``, the memory usage of
            the index is the first item in the output.
        unit : str, default = "B"
            Unit to return. One of {'B', 'KB', 'MB', 'GB'}.

        Returns
        -------
        Series
            A Series whose index is the original column names and whose values
            is the memory usage of each column in bytes.

        See Also
        --------
        arkouda.pdarrayclass.nbytes
        arkouda.index.Index.memory_usage
        arkouda.index.MultiIndex.memory_usage
        arkouda.series.Series.memory_usage

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.connect()
        >>> dtypes = [ak.int64, ak.float64,  ak.bool]
        >>> data = dict([(str(t), ak.ones(5000, dtype=ak.int64).astype(t)) for t in dtypes])
        >>> df = ak.DataFrame(data)
        >>> display(df.head())

        +----+---------+-----------+--------+
        |    |   int64 |   float64 | bool   |
        +====+=========+===========+========+
        |  0 |       1 |         1 | True   |
        +----+---------+-----------+--------+
        |  1 |       1 |         1 | True   |
        +----+---------+-----------+--------+
        |  2 |       1 |         1 | True   |
        +----+---------+-----------+--------+
        |  3 |       1 |         1 | True   |
        +----+---------+-----------+--------+
        |  4 |       1 |         1 | True   |
        +----+---------+-----------+--------+

        >>> df.memory_usage()

        +---------+-------+
        |         |     0 |
        +=========+=======+
        | Index   | 40000 |
        +---------+-------+
        | int64   | 40000 |
        +---------+-------+
        | float64 | 40000 |
        +---------+-------+
        | bool    |  5000 |
        +---------+-------+

        >>> df.memory_usage(index=False)

        +---------+-------+
        |         |     0 |
        +=========+=======+
        | int64   | 40000 |
        +---------+-------+
        | float64 | 40000 |
        +---------+-------+
        | bool    |  5000 |
        +---------+-------+

        >>> df.memory_usage(unit="KB")

        +---------+----------+
        |         |        0 |
        +=========+==========+
        | Index   | 39.0625  |
        +---------+----------+
        | int64   | 39.0625  |
        +---------+----------+
        | float64 | 39.0625  |
        +---------+----------+
        | bool    |  4.88281 |
        +---------+----------+

        To get the approximate total memory usage:

        >>>  df.memory_usage(index=True).sum()

        """
        from arkouda.util import convert_bytes

        if index:
            sizes = [self.index.memory_usage(unit=unit)]
            ret_index = ["Index"]
        else:
            sizes = []
            ret_index = []

        sizes += [convert_bytes(c.nbytes, unit=unit) for col, c in self.items()]
        ret_index += self.columns.values.copy()

        result = Series(sizes, index=array(ret_index))
        return result

    def memory_usage_info(self, unit="GB"):
        """
        A formatted string representation of the size of this DataFrame.

        Parameters
        ----------
        unit : str, default = "GB"
            Unit to return. One of {'KB', 'MB', 'GB'}.

        Returns
        -------
        str
            A string representation of the number of bytes used by this DataFrame in [unit]s.

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({'col1': ak.arange(1000), 'col2': ak.arange(1000)})
        >>> df.memory_usage_info()
        '0.00 GB'

        >>> df.memory_usage_info(unit="KB")
        '15 KB'

        """
        from arkouda.util import convert_bytes

        data_size = convert_bytes(self.memory_usage(index=True).sum(), unit=unit)

        return "{:.2f} {}".format(data_size, unit)

    def to_pandas(self, datalimit=maxTransferBytes, retain_index=False):
        """
        Send this DataFrame to a pandas DataFrame.

        Parameters
        ----------
        datalimit : int, default=arkouda.client.maxTransferBytes
            The maximum number size, in megabytes to transfer. The requested
            DataFrame will be converted to a pandas DataFrame only if the
            estimated size of the DataFrame does not exceed this value.

        retain_index : bool, default=False
            Normally, to_pandas() creates a new range index object. If you want
            to keep the index column, set this to True.

        Returns
        -------
        pandas.DataFrame
            The result of converting this DataFrame to a pandas DataFrame.

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> ak_df = ak.DataFrame({"A": ak.arange(2), "B": -1 * ak.arange(2)})
        >>> type(ak_df)
        arkouda.dataframe.DataFrame
        >>> display(ak_df)

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   0 |   0 |
        +----+-----+-----+
        |  1 |   1 |  -1 |
        +----+-----+-----+

        >>> import pandas as pd
        >>> pd_df = ak_df.to_pandas()
        >>> type(pd_df)
        pandas.core.frame.DataFrame
        >>> display(pd_df)

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   0 |   0 |
        +----+-----+-----+
        |  1 |   1 |  -1 |
        +----+-----+-----+

        """

        self.update_nrows()

        # Estimate how much memory would be required for this DataFrame
        nbytes = 0
        for key, val in self.items():
            if isinstance(val, pdarray):
                nbytes += (val.dtype).itemsize * self._nrows
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
            val = self[key].values
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

    def to_markdown(self, mode="wt", index=True, tablefmt="grid", storage_options=None, **kwargs):
        r"""
        Print DataFrame in Markdown-friendly format.

        Parameters
        ----------
        mode : str, optional
            Mode in which file is opened, "wt" by default.
        index : bool, optional, default True
            Add index (row) labels.
        tablefmt: str = "grid"
            Table format to call from tablulate:
            https://pypi.org/project/tabulate/
        storage_options: dict, optional
            Extra options that make sense for a particular storage connection,
            e.g. host, port, username, password, etc., if using a URL that will be parsed by fsspec,
            e.g., starting “s3://”, “gcs://”.
            An error will be raised if providing this argument with a non-fsspec URL.
            See the fsspec and backend storage implementation docs for the set
            of allowed keys and values.
        **kwargs
            These parameters will be passed to tabulate.

        Note
        ----
        This function should only be called on small DataFrames as it calls pandas.DataFrame.to_markdown:
        https://pandas.pydata.org/pandas-docs/version/1.2.4/reference/api/pandas.DataFrame.to_markdown.html

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({"animal_1": ["elk", "pig"], "animal_2": ["dog", "quetzal"]})
        >>> print(df.to_markdown())
        +----+------------+------------+
        |    | animal_1   | animal_2   |
        +====+============+============+
        |  0 | elk        | dog        |
        +----+------------+------------+
        |  1 | pig        | quetzal    |
        +----+------------+------------+


        Suppress the index:

        >>> print(df.to_markdown(index = False))
        +------------+------------+
        | animal_1   | animal_2   |
        +============+============+
        | elk        | dog        |
        +------------+------------+
        | pig        | quetzal    |
        +------------+------------+

        """
        return self.to_pandas().to_markdown(
            mode=mode, index=index, tablefmt=tablefmt, storage_options=storage_options, **kwargs
        )

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
            File path to save data.
        index : bool, default=False
            If True, save the index column. By default, do not save the index.
        columns: List, default = None
            List of columns to include in the file. If None, writes out all columns.
        file_type: str (single | distribute), default=distribute
            Whether to save to a single file or distribute across Locales.
        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray.

        Notes
        -----
        This method saves one file per locale of the arkouda server. All
        files are prefixed by the path argument and suffixed by their
        locale number.

        See Also
        ---------
        to_parquet
        load

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> import os.path
        >>> from pathlib import Path
        >>> my_path = os.path.join(os.getcwd(), 'hdf_output')
        >>> Path(my_path).mkdir(parents=True, exist_ok=True)

        >>> df = ak.DataFrame({"A":[1,2],"B":[3,4]})
        >>> df.to_hdf(my_path + "/my_data")
        >>> df.load(my_path + "/my_data")

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   1 |   3 |
        +----+-----+-----+
        |  1 |   2 |   4 |
        +----+-----+-----+

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
        mode: str (truncate | append), default=truncate
            Indicates whether the dataset should truncate the file and write or append
            to the file
            Only used when as_dataset=True
        file_type: str (single | distribute), default=distribute
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
                    "num_cols": len(self.columns.values),
                    "column_names": self.columns.values,
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
        the dataset does not exist it is added.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share.
        index : bool, default=False
            If True, save the index column. By default, do not save the index.
        columns: List, default=None
            List of columns to include in the file. If None, writes out all columns.
        repack: bool, default=True
            HDF5 does not release memory on delete. When True, the inaccessible
            data (that was overwritten) is removed. When False, the data remains, but is
            inaccessible. Setting to false will yield better performance, but will cause
            file sizes to expand.

        Returns
        -------
        str
            Success message if successful.

        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray.

        Notes
        -----
        If file does not contain File_Format attribute to indicate how it was saved,
          the file name is checked for _LOCALE#### to determine if it is distributed.
        If the dataset provided does not exist, it will be added.

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> import os.path
        >>> from pathlib import Path
        >>> my_path = os.path.join(os.getcwd(), 'hdf_output')
        >>> Path(my_path).mkdir(parents=True, exist_ok=True)

        >>> df = ak.DataFrame({"A":[1,2],"B":[3,4]})
        >>> df.to_hdf(my_path + "/my_data")
        >>> df.load(my_path + "/my_data")

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   1 |   3 |
        +----+-----+-----+
        |  1 |   2 |   4 |
        +----+-----+-----+

        >>> df2 = ak.DataFrame({"A":[5,6],"B":[7,8]})
        >>> df2.update_hdf(my_path + "/my_data")
        >>> df.load(my_path + "/my_data")

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   5 |   7 |
        +----+-----+-----+
        |  1 |   6 |   8 |
        +----+-----+-----+

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
            File path to save data.
        index : bool, default=False
            If True, save the index column. By default, do not save the index.
        columns: list
            List of columns to include in the file. If None, writes out all columns.
        compression : str (Optional), default=None
            Provide the compression type to use when writing the file.
            Supported values: snappy, gzip, brotli, zstd, lz4
        convert_categoricals: bool, default=False
            Parquet requires all columns to be the same size and Categoricals
            don't satisfy that requirement.
            If set, write the equivalent Strings in place of any Categorical columns.
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
        to_hdf
        load

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> import os.path
        >>> from pathlib import Path
        >>> my_path = os.path.join(os.getcwd(), 'parquet_output')
        >>> Path(my_path).mkdir(parents=True, exist_ok=True)

        >>> df = ak.DataFrame({"A":[1,2],"B":[3,4]})
        >>> df.to_parquet(my_path + "/my_data")
        >>> df.load(my_path + "/my_data")

        +----+-----+-----+
        |    |   B |   A |
        +====+=====+=====+
        |  0 |   3 |   1 |
        +----+-----+-----+
        |  1 |   4 |   2 |
        +----+-----+-----+

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
        r"""
        Writes DataFrame to CSV file(s). File will contain a column for each column in the DataFrame.
        All CSV Files written by Arkouda include a header denoting data types of the columns.
        Unlike other file formats, CSV files store Strings as their UTF-8 format instead of storing
        bytes as uint(8).

        Parameters
        ----------
        path: str
            The filename prefix to be used for saving files. Files will have _LOCALE#### appended
            when they are written to disk.
        index: bool, default=False
            If True, the index of the DataFrame will be written to the file
            as a column.
        columns: list of str (Optional)
            Column names to assign when writing data.
        col_delim: str, default=","
            Value to be used to separate columns within the file.
            Please be sure that the value used DOES NOT appear in your dataset.
        overwrite: bool, default=False
            If True, any existing files matching your provided prefix_path will
            be overwritten. If False, an error will be returned if existing files are found.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Raised if all datasets are not present in all parquet files or if one or
            more of the specified files do not exist.
        RuntimeError
            Raised if one or more of the specified files cannot be opened.
            If `allow_errors` is true this may be raised if no values are returned
            from the server.
        TypeError
            Raised if we receive an unknown arkouda_type returned from the server.

        Notes
        -----
        - CSV format is not currently supported by load/load_all operations.
        - The column delimiter is expected to be the same for column names and data.
        - Be sure that column delimiters are not found within your data.
        - All CSV files must delimit rows using newline ("\\n") at this time.

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> import os.path
        >>> from pathlib import Path
        >>> my_path = os.path.join(os.getcwd(), 'csv_output')
        >>> Path(my_path).mkdir(parents=True, exist_ok=True)

        >>> df = ak.DataFrame({"A":[1,2],"B":[3,4]})
        >>> df.to_csv(my_path + "/my_data")
        >>> df2 = DataFrame.read_csv(my_path + "/my_data" + "_LOCALE0000")
        >>> display(df2)

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   1 |   3 |
        +----+-----+-----+
        |  1 |   2 |   4 |
        +----+-----+-----+

        """
        from arkouda.io import to_csv

        data = self._prep_data(index=index, columns=columns)
        to_csv(data, path, names=columns, col_delim=col_delim, overwrite=overwrite)

    @classmethod
    def read_csv(cls, filename: str, col_delim: str = ","):
        r"""
        Read the columns of a CSV file into an Arkouda DataFrame.
        If the file contains the appropriately formatted header, typed data will be returned.
        Otherwise, all data will be returned as a Strings objects.

        Parameters
        ----------
        filename: str
            Filename to read data from.
        col_delim: str, default=","
            The delimiter for columns within the data.

        Returns
        -------
        arkouda.dataframe.DataFrame
            Arkouda DataFrame containing the columns from the CSV file.

        Raises
        ------
        ValueError
            Raised if all datasets are not present in all parquet files or if one or
            more of the specified files do not exist.
        RuntimeError
            Raised if one or more of the specified files cannot be opened.
            If `allow_errors` is true this may be raised if no values are returned
            from the server.
        TypeError
            Raised if we receive an unknown arkouda_type returned from the server.

        See Also
        --------
        to_csv

        Notes
        ------
        - CSV format is not currently supported by load/load_all operations.
        - The column delimiter is expected to be the same for column names and data.
        - Be sure that column delimiters are not found within your data.
        - All CSV files must delimit rows using newline ("\\n") at this time.
        - Unlike other file formats, CSV files store Strings as their UTF-8 format instead of storing
        bytes as uint(8).

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> import os.path
        >>> from pathlib import Path
        >>> my_path = os.path.join(os.getcwd(), 'csv_output','my_data')
        >>> Path(my_path).mkdir(parents=True, exist_ok=True)

        >>> df = ak.DataFrame({"A":[1,2],"B":[3,4]})
        >>> df.to_csv(my_path)
        >>> df2 = DataFrame.read_csv(my_path + "_LOCALE0000")
        >>> display(df2)

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   1 |   3 |
        +----+-----+-----+
        |  1 |   2 |   4 |
        +----+-----+-----+

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
            File path to save data.
        index : bool, default=False
            If True, save the index column. By default, do not save the index.
        columns: list, default=None
            List of columns to include in the file. If None, writes out all columns.
        file_format : str, default='HDF5'
            'HDF5' or 'Parquet'. Defaults to 'HDF5'
        file_type : str, default=distribute
            "single" or "distribute"
            If single, will right a single file to locale 0.
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

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> import os.path
        >>> from pathlib import Path
        >>> my_path = os.path.join(os.getcwd(), 'hdf5_output')
        >>> Path(my_path).mkdir(parents=True, exist_ok=True)

        >>> df = ak.DataFrame({"A": ak.arange(5), "B": -1 * ak.arange(5)})
        >>> df.save(my_path + '/my_data', file_type="single")
        >>> df.load(my_path + '/my_data')

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   0 |   0 |
        +----+-----+-----+
        |  1 |   1 |  -1 |
        +----+-----+-----+
        |  2 |   2 |  -2 |
        +----+-----+-----+
        |  3 |   3 |  -3 |
        +----+-----+-----+
        |  4 |   4 |  -4 |
        +----+-----+-----+

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
        Load dataframe from file.
        file_format needed for consistency with other load functions.

        Parameters
        ----------
        prefix_path : str
            The prefix path for the data.

        file_format : string, default = "INFER"

        Returns
        -------
        arkouda.dataframe.DataFrame
            A dataframe loaded from the prefix_path.

        Examples
        --------

        To store data in <my_dir>/my_data_LOCALE0000,
        use "<my_dir>/my_data" as the prefix.

        >>> import arkouda as ak
        >>> ak.connect()
        >>> import os.path
        >>> from pathlib import Path
        >>> my_path = os.path.join(os.getcwd(), 'hdf5_output','my_data')
        >>> Path(my_path).mkdir(parents=True, exist_ok=True)
        >>> df = ak.DataFrame({"A": ak.arange(5), "B": -1 * ak.arange(5)})
        >>> df.save(my_path, file_type="distribute")
        >>> df.load(my_path)

        +----+-----+-----+
        |    |   A |   B |
        +====+=====+=====+
        |  0 |   0 |   0 |
        +----+-----+-----+
        |  1 |   1 |  -1 |
        +----+-----+-----+
        |  2 |   2 |  -2 |
        +----+-----+-----+
        |  3 |   3 |  -3 |
        +----+-----+-----+
        |  4 |   4 |  -4 |
        +----+-----+-----+

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
        return df if filetype == "HDF5" else df[df.columns.values[::-1]]

    def argsort(self, key, ascending=True):
        """
        Return the permutation that sorts the dataframe by `key`.

        Parameters
        ----------
        key : str
            The key to sort on.
        ascending : bool, default = True
            If true, sort the key in ascending order.
            Otherwise, sort the key in descending order.

        Returns
        -------
        arkouda.pdarrayclass.pdarray
            The permutation array that sorts the data on `key`.

        See Also
        --------
        coargsort

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({'col1': [1.1, 3.1, 2.1], 'col2': [6, 5, 4]})
        >>> display(df)

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |    1.1 |      6 |
        +----+--------+--------+
        |  1 |    3.1 |      5 |
        +----+--------+--------+
        |  2 |    2.1 |      4 |
        +----+--------+--------+

        >>> df.argsort('col1')
        array([0 2 1])
        >>> sorted_df1 = df[df.argsort('col1')]
        >>> display(sorted_df1)

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |    1.1 |      6 |
        +----+--------+--------+
        |  1 |    2.1 |      4 |
        +----+--------+--------+
        |  2 |    3.1 |      5 |
        +----+--------+--------+

        >>> df.argsort('col2')
        array([2 1 0])
        >>> sorted_df2 = df[df.argsort('col2')]
        >>> display(sorted_df2)

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |    2.1 |      4 |
        +----+--------+--------+
        |  1 |    3.1 |      5 |
        +----+--------+--------+
        |  2 |    1.1 |      6 |
        +----+--------+--------+

        """

        if self._empty:
            return array([], dtype=akint64)
        if ascending:
            return argsort(self.data[key])
        else:
            if isinstance(self.data[key], pdarray) and self.data[key].dtype in (
                akint64,
                akfloat64,
            ):
                return argsort(-self.data[key])
            else:
                return argsort(self.data[key])[arange(self._nrows - 1, -1, -1)]

    def coargsort(self, keys, ascending=True):
        """
        Return the permutation that sorts the dataframe by `keys`.

        Note: Sorting using Strings may not yield correct sort order.

        Parameters
        ----------
        keys : list of str
            The keys to sort on.

        Returns
        -------
        arkouda.pdarrayclass.pdarray
            The permutation array that sorts the data on `keys`.

        Example
        -------

        >>> df = ak.DataFrame({'col1': [2, 2, 1], 'col2': [3, 4, 3], 'col3':[5, 6, 7]})
        >>> display(df)

        +----+--------+--------+--------+
        |    |   col1 |   col2 |   col3 |
        +====+========+========+========+
        |  0 |      2 |      3 |      5 |
        +----+--------+--------+--------+
        |  1 |      2 |      4 |      6 |
        +----+--------+--------+--------+
        |  2 |      1 |      3 |      7 |
        +----+--------+--------+--------+

        >>> df.coargsort(['col1', 'col2'])
        array([2 0 1])
        >>>


        """

        if self._empty:
            return array([], dtype=akint64)
        arrays = []
        for key in keys:
            arrays.append(self[key].values)
        i = coargsort(arrays)
        if not ascending:
            i = i[arange(self._nrows - 1, -1, -1)]
        return i

    def _reindex(self, idx):
        if isinstance(self.index, MultiIndex):
            new_index = MultiIndex(self.index[idx].values, name=self.index.name, names=self.index.names)
        elif isinstance(self.index, Index):
            new_index = Index(self.index[idx], name=self.index.name)
        else:
            new_index = Index(self.index[idx])

        return DataFrame(self.iloc[idx], index=new_index)

    def sort_index(self, ascending=True):
        """
        Sort the DataFrame by indexed columns.

        Note: Fails on sort order of arkouda.strings.Strings columns when multiple columns being sorted.

        Parameters
        ----------
        ascending : bool, default = True
            Sort values in ascending (default) or descending order.

        Example
        -------

        >>> df = ak.DataFrame({'col1': [1.1, 3.1, 2.1], 'col2': [6, 5, 4]},
        ...          index = Index(ak.array([2,0,1]), name="idx"))

        >>> display(df)

        +----+--------+--------+
        | idx|   col1 |   col2 |
        +====+========+========+
        |  0 |    1.1 |      6 |
        +----+--------+--------+
        |  1 |    3.1 |      5 |
        +----+--------+--------+
        |  2 |    2.1 |      4 |
        +----+--------+--------+

        >>> df.sort_index()

        +----+--------+--------+
        | idx|   col1 |   col2 |
        +====+========+========+
        |  0 |    3.1 |      5 |
        +----+--------+--------+
        |  1 |    2.1 |      4 |
        +----+--------+--------+
        |  2 |    1.1 |      6 |
        +----+--------+--------+

        """

        idx = self.index.argsort(ascending=ascending)

        return self._reindex(idx)

    def sort_values(self, by=None, ascending=True):
        """
        Sort the DataFrame by one or more columns.

        If no column is specified, all columns are used.

        Note: Fails on order of arkouda.strings.Strings columns when multiple columns being sorted.

        Parameters
        ----------
        by : str or list/tuple of str, default = None
            The name(s) of the column(s) to sort by.
        ascending : bool, default = True
            Sort values in ascending (default) or descending order.

        See Also
        --------
        apply_permutation

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({'col1': [2, 2, 1], 'col2': [3, 4, 3], 'col3':[5, 6, 7]})
        >>> display(df)

        +----+--------+--------+--------+
        |    |   col1 |   col2 |   col3 |
        +====+========+========+========+
        |  0 |      2 |      3 |      5 |
        +----+--------+--------+--------+
        |  1 |      2 |      4 |      6 |
        +----+--------+--------+--------+
        |  2 |      1 |      3 |      7 |
        +----+--------+--------+--------+

        >>> df.sort_values()

        +----+--------+--------+--------+
        |    |   col1 |   col2 |   col3 |
        +====+========+========+========+
        |  0 |      1 |      3 |      7 |
        +----+--------+--------+--------+
        |  1 |      2 |      3 |      5 |
        +----+--------+--------+--------+
        |  2 |      2 |      4 |      6 |
        +----+--------+--------+--------+

        >>> df.sort_values("col3")

        +----+--------+--------+--------+
        |    |   col1 |   col2 |   col3 |
        +====+========+========+========+
        |  0 |      1 |      3 |      7 |
        +----+--------+--------+--------+
        |  1 |      2 |      3 |      5 |
        +----+--------+--------+--------+
        |  2 |      2 |      4 |      6 |
        +----+--------+--------+--------+

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
        return self.iloc[i]

    def apply_permutation(self, perm):
        """
        Apply a permutation to an entire DataFrame.  The operation is done in
        place and the original DataFrame will be modified.

        This may be useful if you want to unsort an DataFrame, or even to
        apply an arbitrary permutation such as the inverse of a sorting
        permutation.

        Parameters
        ----------
        perm : pdarray
            A permutation array. Should be the same size as the data
            arrays, and should consist of the integers [0,size-1] in
            some order. Very minimal testing is done to ensure this
            is a permutation.

        Returns
        -------
        None

        See Also
        --------
        sort

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      1 |      4 |
        +----+--------+--------+
        |  1 |      2 |      5 |
        +----+--------+--------+
        |  2 |      3 |      6 |
        +----+--------+--------+

        >>> perm_arry = ak.array([0, 2, 1])
        >>> df.apply_permutation(perm_arry)
        >>> display(df)

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      1 |      4 |
        +----+--------+--------+
        |  1 |      3 |      6 |
        +----+--------+--------+
        |  2 |      2 |      5 |
        +----+--------+--------+

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
        keys : str or list of str
            The names of the columns to group by.
        low : int, default=1
            The lowest value count.
        high : int, default=None
            The highest value count, default to unlimited.

        Returns
        -------
        arkouda.pdarrayclass.pdarray
            An array of boolean values for qualified rows in this DataFrame.

        Example
        -------

        >>> df = ak.DataFrame({'col1': [1, 2, 2, 2, 3, 3], 'col2': [4, 5, 6, 7, 8, 9]})
        >>> display(df)

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      1 |      4 |
        +----+--------+--------+
        |  1 |      2 |      5 |
        +----+--------+--------+
        |  2 |      2 |      6 |
        +----+--------+--------+
        |  3 |      2 |      7 |
        +----+--------+--------+
        |  4 |      3 |      8 |
        +----+--------+--------+
        |  5 |      3 |      9 |
        +----+--------+--------+

        >>> df.filter_by_range("col1", low=1, high=2)
        array([True False False False True True])

        >>> filtered_df = df[df.filter_by_range("col1", low=1, high=2)]
        >>> display(filtered_df)

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      1 |      4 |
        +----+--------+--------+
        |  1 |      3 |      8 |
        +----+--------+--------+
        |  2 |      3 |      9 |
        +----+--------+--------+

        """
        if isinstance(keys, str):
            keys = [keys]
        gb = self.GroupBy(keys, use_series=False)
        vals, cts = gb.size()
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
        deep : bool, default=True
            When True, return a deep copy. Otherwise, return a shallow copy.

        Returns
        -------
        arkouda.dataframe.DataFrame
            A deep or shallow copy according to caller specification.

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> display(df)

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      1 |      3 |
        +----+--------+--------+
        |  1 |      2 |      4 |
        +----+--------+--------+

        >>> df_deep = df.copy(deep=True)
        >>> df_deep['col1'] +=1
        >>> display(df)

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      1 |      3 |
        +----+--------+--------+
        |  1 |      2 |      4 |
        +----+--------+--------+

        >>> df_shallow = df.copy(deep=False)
        >>> df_shallow['col1'] +=1
        >>> display(df)

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      2 |      3 |
        +----+--------+--------+
        |  1 |      3 |      4 |
        +----+--------+--------+

        """
        if deep:
            res = DataFrame()
            res._size = self._nrows
            res._bytes = self._bytes
            res._empty = self._empty

            for col in self._columns:
                res[col] = self[col].iloc[:]
            # if this is not a slice, renaming indexes with update both
            res._set_index(Index(self.index.index[:]))
            return res
        else:
            return DataFrame(self)

    def groupby(self, keys, use_series=True, as_index=True, dropna=True):
        """
        Group the dataframe by a column or a list of columns.  Alias for GroupBy.

        Parameters
        ----------
        keys : str or list of str
            An (ordered) list of column names or a single string to group by.
        use_series : bool, default=True
            If True, returns an arkouda.dataframe.DataFrameGroupBy object.
            Otherwise an arkouda.groupbyclass.GroupBy object.
        as_index: bool, default=True
            If True, groupby columns will be set as index
            otherwise, the groupby columns will be treated as DataFrame columns.
        dropna : bool, default=True
            If True, and the groupby keys contain NaN values,
            the NaN values together with the corresponding row will be dropped.
            Otherwise, the rows corresponding to NaN values will be kept.
        Returns
        -------
        arkouda.dataframe.DataFrameGroupBy or arkouda.groupbyclass.GroupBy
            If use_series = True, returns an arkouda.dataframe.DataFrameGroupBy object.
            Otherwise returns an arkouda.groupbyclass.GroupBy object.

        See Also
        --------
        arkouda.GroupBy

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({'col1': [1.0, 1.0, 2.0, np.nan], 'col2': [4, 5, 6, 7]})
        >>> df

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      1 |      4 |
        +----+--------+--------+
        |  1 |      1 |      5 |
        +----+--------+--------+
        |  2 |      2 |      6 |
        +----+--------+--------+
        |  3 |    nan |      7 |
        +----+--------+--------+

        >>> df.GroupBy("col1")
        <arkouda.groupbyclass.GroupBy at 0x7f2cf23e10c0>
        >>> df.GroupBy("col1").size()
        (array([1.00000000000000000 2.00000000000000000]), array([2 1]))

        >>> df.GroupBy("col1",use_series=True)
        col1
        1.0    2
        2.0    1
        dtype: int64
        >>> df.GroupBy("col1",use_series=True, as_index = False).size()

        +----+--------+--------+
        |    |   col1 |   size |
        +====+========+========+
        |  0 |      1 |      2 |
        +----+--------+--------+
        |  1 |      2 |      1 |
        +----+--------+--------+

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
        arkouda.dataframe.DataFrame
            Arkouda DataFrame of booleans showing whether each element in the DataFrame is
            contained in values.

        See Also
        ________
        ak.Series.isin

        Notes
        _____
        - Pandas supports values being an iterable type. In arkouda, we replace this with pdarray.
        - Pandas supports ~ operations. Currently, ak.DataFrame does not support this.

        Examples
        ________

        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({'col_A': ak.array([7, 3]), 'col_B':ak.array([1, 9])})
        >>> display(df)

        +----+---------+---------+
        |    |   col_A |   col_B |
        +====+=========+=========+
        |  0 |       7 |       1 |
        +----+---------+---------+
        |  1 |       3 |       9 |
        +----+---------+---------+

        When `values` is a pdarray, check every value in the DataFrame to determine if
        it exists in values.

        >>> df.isin(ak.array([0, 1]))

        +----+---------+---------+
        |    |   col_A |   col_B |
        +====+=========+=========+
        |  0 |       0 |       1 |
        +----+---------+---------+
        |  1 |       0 |       0 |
        +----+---------+---------+

        When `values` is a dict, the values in the dict are passed to check the column
        indicated by the key.

        >>> df.isin({'col_A': ak.array([0, 3])})

        +----+---------+---------+
        |    |   col_A |   col_B |
        +====+=========+=========+
        |  0 |       0 |       0 |
        +----+---------+---------+
        |  1 |       1 |       0 |
        +----+---------+---------+

        When `values` is a Series, each column is checked if values is present positionally.
        This means that for `True` to be returned, the indexes must be the same.

        >>> i = ak.Index(ak.arange(2))
        >>> s = ak.Series(data=[3, 9], index=i)
        >>> df.isin(s)

        +----+---------+---------+
        |    |   col_A |   col_B |
        +====+=========+=========+
        |  0 |       0 |       0 |
        +----+---------+---------+
        |  1 |       0 |       1 |
        +----+---------+---------+

        When `values` is a DataFrame, the index and column must match.
        Note that 9 is not found because the column name does not match.

        >>> other_df = ak.DataFrame({'col_A':ak.array([7, 3]), 'col_C':ak.array([0, 9])})
        >>> df.isin(other_df)

        +----+---------+---------+
        |    |   col_A |   col_B |
        +====+=========+=========+
        |  0 |       1 |       0 |
        +----+---------+---------+
        |  1 |       1 |       0 |
        +----+---------+---------+

        """
        if isinstance(values, pdarray):
            # flatten the DataFrame so single in1d can be used.
            flat_in1d = in1d(concatenate(list(self.data.values())), values)
            segs = concatenate(
                [
                    array([0]),
                    cumsum(array([self.data[col].size for col in self.columns.values])),
                ]
            )
            df_def = {col: flat_in1d[segs[i] : segs[i + 1]] for i, col in enumerate(self.columns.values)}
        elif isinstance(values, Dict):
            # key is column name, val is the list of values to check
            df_def = {
                col: (
                    in1d(self.data[col], values[col])
                    if col in values.keys()
                    else zeros(self._nrows, dtype=akbool)
                )
                for col in self.columns.values
            }
        elif isinstance(values, DataFrame) or (
            isinstance(values, Series) and isinstance(values.index, Index)
        ):
            # create the dataframe with all false
            df_def = {col: zeros(self._nrows, dtype=akbool) for col in self.columns.values}
            # identify the indexes in both
            rows_self, rows_val = intersect(self.index.index, values.index.index, unique=True)

            # used to sort the rows with only the indexes in both
            sort_self = self.index[rows_self].argsort()
            sort_val = values.index[rows_val].argsort()
            # update values in columns that exist in both. only update the rows whose indexes match

            for col in self.columns.values:
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

    def count(self, axis: Union[int, str] = 0, numeric_only=False) -> Series:
        """
        Count non-NA cells for each column or row.

        The values np.NaN are considered NA.

        Parameters
        __________
        axis : {0 or 'index', 1 or 'columns'}, default 0
            If 0 or ‘index’ counts are generated for each column.
            If 1 or ‘columns’ counts are generated for each row.

        numeric_only: bool = False
            Include only float, int or boolean data.

        Returns
        _______
        arkouda.series.Series
            For each column/row the number of non-NA/null entries.

        Raises
        ------
        ValueError
            Raised if axis is not 0, 1, 'index', or 'columns'.

        See Also
        ________
        GroupBy.count()

        Examples
        ________

        >>> import arkouda as ak
        >>> ak.connect()
        >>> import numpy as np
        >>> df = ak.DataFrame({'col_A': ak.array([7, np.nan]), 'col_B':ak.array([1, 9])})
        >>> display(df)

        +----+---------+---------+
        |    |   col_A |   col_B |
        +====+=========+=========+
        |  0 |       7 |       1 |
        +----+---------+---------+
        |  1 |     nan |       9 |
        +----+---------+---------+

        >>> df.count()
        col_A    1
        col_B    2
        dtype: int64

        >>> df = ak.DataFrame({'col_A': ak.array(["a","b","c"]), 'col_B':ak.array([1, np.nan, np.nan])})
        >>> display(df)

        +----+---------+---------+
        |    | col_A   |   col_B |
        +====+=========+=========+
        |  0 | a       |       1 |
        +----+---------+---------+
        |  1 | b       |     nan |
        +----+---------+---------+
        |  2 | c       |     nan |
        +----+---------+---------+

        >>> df.count()
        col_A    3
        col_B    1
        dtype: int64

        >>> df.count(numeric_only=True)
        col_B    1
        dtype: int64

        >>> df.count(axis=1)
        0    2
        1    1
        2    1
        dtype: int64

        """
        from arkouda import full, isnan
        from arkouda.util import is_numeric

        if (isinstance(axis, int) and axis == 0) or (isinstance(axis, str) and axis == "index"):
            index_values_list = []
            count_values_list = []
            for col in self.columns:
                if is_numeric(self[col].values):
                    index_values_list.append(col)
                    count_values_list.append((~isnan(self[col].values)).sum())
                elif not numeric_only or self[col].dtype == bool:
                    index_values_list.append(col)
                    # Non-numeric columns do not have NaN values.
                    count_values_list.append(self[col].size)
            return Series(array(count_values_list), index=Index(array(index_values_list)))
        elif (isinstance(axis, int) and axis == 1) or (isinstance(axis, str) and axis == "columns"):
            first = True
            count_values = arange(0)
            for col in self.columns:
                if is_numeric(self[col].values):
                    if first:
                        count_values = akcast(~isnan(self[col].values), dt="int64")
                        first = False
                    else:
                        count_values += ~isnan(self[col].values)
                elif not numeric_only or self[col].dtype == bool:
                    if first:
                        count_values = full(self.index.size, 1, dtype=akint64)
                        first = False
                    else:
                        count_values += 1
                if first:
                    count_values = full(self.index.size, 0, dtype=akint64)
            if self.index is not None:
                idx = self.index[:]
                return Series(array(count_values), index=idx)
            else:
                return Series(array(count_values))
        else:
            raise ValueError(f"No axis named {axis} for object type DataFrame")

    def corr(self) -> DataFrame:
        """
        Return new DataFrame with pairwise correlation of columns.

        Returns
        -------
        arkouda.dataframe.DataFrame
            Arkouda DataFrame containing correlation matrix of all columns.

        Raises
        ------
        RuntimeError
            Raised if there's a server-side error thrown.

        See Also
        --------
        pdarray.corr

        Notes
        -----
        Generates the correlation matrix using Pearson R for all columns.

        Attempts to convert to numeric values where possible for inclusion in the matrix.

        Example
        -------

        >>> df = ak.DataFrame({'col1': [1, 2], 'col2': [-1, -2]})
        >>> display(df)

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      1 |     -1 |
        +----+--------+--------+
        |  1 |      2 |     -2 |
        +----+--------+--------+

        >>> corr = df.corr()

        +------+--------+--------+
        |      |   col1 |   col2 |
        +======+========+========+
        | col1 |      1 |     -1 |
        +------+--------+--------+
        | col2 |     -1 |      1 |
        +------+--------+--------+

        """

        def numeric_help(d):
            if isinstance(d, Strings):
                d = Categorical(d)
            return d if isinstance(d, pdarray) else d.codes

        args = {
            "size": len(self.columns.values),
            "columns": self.columns.values,
            "data_names": [numeric_help(self[c].values) for c in self.columns.values],
        }

        ret_dict = json.loads(generic_msg(cmd="corrMatrix", args=args))
        return DataFrame(
            {c: create_pdarray(ret_dict[c]) for c in self.columns.values},
            index=array(self.columns.values),
        )

    @typechecked
    def merge(
        self,
        right: DataFrame,
        on: Optional[Union[str, List[str]]] = None,
        how: str = "inner",
        left_suffix: str = "_x",
        right_suffix: str = "_y",
        convert_ints: bool = True,
        sort: bool = True,
    ) -> DataFrame:
        r"""
        Merge Arkouda DataFrames with a database-style join.
        The resulting dataframe contains rows from both DataFrames as specified by
        the merge condition (based on the "how" and "on" parameters).

        Based on pandas merge functionality.
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html

        Parameters
        ----------
        right: DataFrame
            The Right DataFrame to be joined.
        on: Optional[Union[str, List[str]]] = None
            The name or list of names of the DataFrame column(s) to join on.
            If on is None, this defaults to the intersection of the columns in both DataFrames.
        how:  {"inner", "left", "right}, default = "inner"
            The merge condition.
            Must be "inner", "left", or "right".
        left_suffix: str, default = "_x"
            A string indicating the suffix to add to columns from the left dataframe for overlapping
            column names in both left and right. Defaults to "_x". Only used when how is "inner".
        right_suffix: str, default = "_y"
            A string indicating the suffix to add to columns from the right dataframe for overlapping
            column names in both left and right. Defaults to "_y". Only used when how is "inner".
        convert_ints: bool = True
            If True, convert columns with missing int values (due to the join) to float64.
            This is to match pandas.
            If False, do not convert the column dtypes.
            This has no effect when how = "inner".
        sort: bool = True
            If True, DataFrame is returned sorted by "on".
            Otherwise, the DataFrame is not sorted.
        Returns
        -------
        arkouda.dataframe.DataFrame
            Joined Arkouda DataFrame.

        Note
        ----
        Multiple column joins are only supported for integer columns.

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> left_df = ak.DataFrame({'col1': ak.arange(5), 'col2': -1 * ak.arange(5)})
        >>> display(left_df)

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      0 |      0 |
        +----+--------+--------+
        |  1 |      1 |     -1 |
        +----+--------+--------+
        |  2 |      2 |     -2 |
        +----+--------+--------+
        |  3 |      3 |     -3 |
        +----+--------+--------+
        |  4 |      4 |     -4 |
        +----+--------+--------+

        >>> right_df = ak.DataFrame({'col1': 2 * ak.arange(5), 'col2': 2 * ak.arange(5)})
        >>> display(right_df)

        +----+--------+--------+
        |    |   col1 |   col2 |
        +====+========+========+
        |  0 |      0 |      0 |
        +----+--------+--------+
        |  1 |      2 |      2 |
        +----+--------+--------+
        |  2 |      4 |      4 |
        +----+--------+--------+
        |  3 |      6 |      6 |
        +----+--------+--------+
        |  4 |      8 |      8 |
        +----+--------+--------+

        >>> left_df.merge(right_df, on = "col1")

        +----+--------+----------+----------+
        |    |   col1 |   col2_x |   col2_y |
        +====+========+==========+==========+
        |  0 |      0 |        0 |        0 |
        +----+--------+----------+----------+
        |  1 |      2 |       -2 |        2 |
        +----+--------+----------+----------+
        |  2 |      4 |       -4 |        4 |
        +----+--------+----------+----------+

        >>> left_df.merge(right_df, on = "col1", how = "left")

        +----+--------+----------+----------+
        |    |   col1 |   col2_y |   col2_x |
        +====+========+==========+==========+
        |  0 |      0 |        0 |        0 |
        +----+--------+----------+----------+
        |  1 |      1 |      nan |       -1 |
        +----+--------+----------+----------+
        |  2 |      2 |        2 |       -2 |
        +----+--------+----------+----------+
        |  3 |      3 |      nan |       -3 |
        +----+--------+----------+----------+
        |  4 |      4 |        4 |       -4 |
        +----+--------+----------+----------+

        >>> left_df.merge(right_df, on = "col1", how = "right")

        +----+--------+----------+----------+
        |    |   col1 |   col2_x |   col2_y |
        +====+========+==========+==========+
        |  0 |      0 |        0 |        0 |
        +----+--------+----------+----------+
        |  1 |      2 |       -2 |        2 |
        +----+--------+----------+----------+
        |  2 |      4 |       -4 |        4 |
        +----+--------+----------+----------+
        |  3 |      6 |      nan |        6 |
        +----+--------+----------+----------+
        |  4 |      8 |      nan |        8 |
        +----+--------+----------+----------+

        >>> left_df.merge(right_df, on = "col1", how = "outer")

        +----+--------+----------+----------+
        |    |   col1 |   col2_y |   col2_x |
        +====+========+==========+==========+
        |  0 |      0 |        0 |        0 |
        +----+--------+----------+----------+
        |  1 |      1 |      nan |       -1 |
        +----+--------+----------+----------+
        |  2 |      2 |        2 |       -2 |
        +----+--------+----------+----------+
        |  3 |      3 |      nan |       -3 |
        +----+--------+----------+----------+
        |  4 |      4 |        4 |       -4 |
        +----+--------+----------+----------+
        |  5 |      6 |        6 |      nan |
        +----+--------+----------+----------+
        |  6 |      8 |        8 |      nan |
        +----+--------+----------+----------+

        """
        return merge(
            self, right, on, how, left_suffix, right_suffix, convert_ints=convert_ints, sort=sort
        )

    @typechecked
    def isna(self) -> DataFrame:
        """
        Detect missing values.

        Return a boolean same-sized object indicating if the values are NA.
        numpy.NaN values get mapped to True values.
        Everything else gets mapped to False values.

        Returns
        -------
        arkouda.dataframe.DataFrame
            Mask of bool values for each element in DataFrame
            that indicates whether an element is an NA value.

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.connect()
        >>> import numpy as np
        >>> df = ak.DataFrame({"A": [np.nan, 2, 2, 3], "B": [3, np.nan, 5, 6],
        ...          "C": [1, np.nan, 2, np.nan], "D":["a","b","c","d"]})
        >>> display(df)

        +----+-----+-----+-----+-----+
        |    |   A |   B |   C | D   |
        +====+=====+=====+=====+=====+
        |  0 | nan |   3 |   1 | a   |
        +----+-----+-----+-----+-----+
        |  1 |   2 | nan | nan | b   |
        +----+-----+-----+-----+-----+
        |  2 |   2 |   5 |   2 | c   |
        +----+-----+-----+-----+-----+
        |  3 |   3 |   6 | nan | d   |
        +----+-----+-----+-----+-----+

        >>> df.isna()
               A      B      C      D
        0   True  False  False  False
        1  False   True   True  False
        2  False  False  False  False
        3  False  False   True  False (4 rows x 4 columns)

        """
        from arkouda import full, isnan
        from arkouda.util import is_numeric

        def is_nan_col(col: str):
            if is_numeric(self[col].values):
                return isnan(self[col].values)
            else:
                return full(self.shape[0], False, dtype=akbool)

        data = {col: is_nan_col(col) for col in self.columns.values}
        return DataFrame(data)

    @typechecked
    def notna(self) -> DataFrame:
        """
        Detect existing (non-missing) values.

        Return a boolean same-sized object indicating if the values are not NA.
        numpy.NaN values get mapped to False values.

        Returns
        -------
        arkouda.dataframe.DataFrame
            Mask of bool values for each element in DataFrame
            that indicates whether an element is not an NA value.

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.connect()
        >>> import numpy as np
        >>> df = ak.DataFrame({"A": [np.nan, 2, 2, 3], "B": [3, np.nan, 5, 6],
        ...          "C": [1, np.nan, 2, np.nan], "D":["a","b","c","d"]})
        >>> display(df)

        +----+-----+-----+-----+-----+
        |    |   A |   B |   C | D   |
        +====+=====+=====+=====+=====+
        |  0 | nan |   3 |   1 | a   |
        +----+-----+-----+-----+-----+
        |  1 |   2 | nan | nan | b   |
        +----+-----+-----+-----+-----+
        |  2 |   2 |   5 |   2 | c   |
        +----+-----+-----+-----+-----+
        |  3 |   3 |   6 | nan | d   |
        +----+-----+-----+-----+-----+

        >>> df.notna()
               A      B      C     D
        0  False   True   True  True
        1   True  False  False  True
        2   True   True   True  True
        3   True   True  False  True (4 rows x 4 columns)

        """
        from arkouda import full, isnan
        from arkouda.util import is_numeric

        def not_nan_col(col: str):
            if is_numeric(self[col].values):
                return ~isnan(self[col].values)
            else:
                return full(self.shape[0], True, dtype=akbool)

        data = {col: not_nan_col(col) for col in self.columns.values}
        return DataFrame(data)

    @typechecked
    def any(self, axis=0) -> Union[Series, bool]:
        """
        Return whether any element is True, potentially over an axis.

        Returns False unless there is at least one element along a Dataframe axis that is True.

        Currently, will ignore any columns that are not type bool.
        This is equivalent to the pandas option bool_only=True.

        Parameters
        ----------
        axis: {0 or ‘index’, 1 or ‘columns’, None}, default = 0

            Indicate which axis or axes should be reduced.

            0 / ‘index’ : reduce the index, return a Series whose index is the original column labels.

            1 / ‘columns’ : reduce the columns, return a Series whose index is the original index.

            None : reduce all axes, return a scalar.

        Returns
        -------
        arkouda.series.Series or bool

        Raises
        ------
        ValueError
            Raised if axis does not have a value in {0 or ‘index’, 1 or ‘columns’, None}.

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({"A":[True,True,True,False],"B":[True,True,True,False],
        ...          "C":[True,False,True,False],"D":[False,False,False,False]})

        +----+---------+---------+---------+---------+
        |    |   A     |   B     |   C     |   D     |
        +====+=========+=========+=========+=========+
        |  0 |   True  |   True  |   True  |   False |
        +----+---------+---------+---------+---------+
        |  1 |   True  |   True  |   False |   False |
        +----+---------+---------+---------+---------+
        |  2 |   True  |   True  |   True  |   False |
        +----+---------+---------+---------+---------+
        |  3 |   False |   False |   False |   False |
        +----+---------+---------+---------+---------+

        >>> df.any(axis=0)
        A     True
        B     True
        C     True
        D    False
        dtype: bool
        >>> df.any(axis=1)
        0     True
        1     True
        2     True
        3    False
        dtype: bool
        >>> df.any(axis=None)
        True

        """
        from arkouda import any as akany
        from arkouda import array, full

        if self.empty:
            if axis is None:
                return False
            else:
                return Series(array([], dtype=bool))

        bool_cols = [col for col in self.columns.values if self.dtypes[col] == "bool"]
        if (isinstance(axis, int) and axis == 0) or (isinstance(axis, str) and axis == "index"):
            return Series(
                array([akany(self[col].values) for col in bool_cols]),
                index=Index(bool_cols),
            )
        elif (isinstance(axis, int) and axis == 1) or (isinstance(axis, str) and axis == "columns"):
            mask = None
            first = True
            for col in bool_cols:
                if first:
                    mask = self[col]
                    first = False
                else:
                    mask |= self[col]
            if first:
                mask = full(self.shape[0], False, dtype=bool)
            return Series(mask, index=self.index.values[:])
        elif axis is None:
            return any([akany(self[col].values) for col in bool_cols])
        else:
            raise ValueError("axis must have value 0, 1, 'index', 'columns', or None.")

    @typechecked
    def all(self, axis=0) -> Union[Series, bool]:
        """
        Return whether all elements are True, potentially over an axis.

        Returns True unless there at least one element along a Dataframe axis that is False.

        Currently, will ignore any columns that are not type bool.
        This is equivalent to the pandas option bool_only=True.

        Parameters
        ----------
        axis: {0 or ‘index’, 1 or ‘columns’, None}, default = 0

            Indicate which axis or axes should be reduced.

            0 / ‘index’ : reduce the index, return a Series whose index is the original column labels.

            1 / ‘columns’ : reduce the columns, return a Series whose index is the original index.

            None : reduce all axes, return a scalar.

        Returns
        -------
        arkouda.series.Series or bool

        Raises
        ------
        ValueError
            Raised if axis does not have a value in {0 or ‘index’, 1 or ‘columns’, None}.

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.connect()
        >>> df = ak.DataFrame({"A":[True,True,True,False],"B":[True,True,True,False],
        ...          "C":[True,False,True,False],"D":[True,True,True,True]})

        +----+---------+---------+---------+--------+
        |    |   A     |   B     |   C     |   D    |
        +====+=========+=========+=========+========+
        |  0 |   True  |   True  |   True  |   True |
        +----+---------+---------+---------+--------+
        |  1 |   True  |   True  |   False |   True |
        +----+---------+---------+---------+--------+
        |  2 |   True  |   True  |   True  |   True |
        +----+---------+---------+---------+--------+
        |  3 |   False |   False |   False |   True |
        +----+---------+---------+---------+--------+

        >>> df.all(axis=0)
        A    False
        B    False
        C    False
        D     True
        dtype: bool
        >>> df.all(axis=1)
        0     True
        1    False
        2     True
        3    False
        dtype: bool
        >>> df.all(axis=None)
        False
        """
        from arkouda import all as akall
        from arkouda import array, full

        if self.empty:
            if axis is None:
                return True
            else:
                return Series(array([], dtype=bool))

        bool_cols = [col for col in self.columns.values if self.dtypes[col] == "bool"]
        if (isinstance(axis, int) and axis == 0) or (isinstance(axis, str) and axis == "index"):
            return Series(
                array([akall(self[col].values) for col in bool_cols]),
                index=Index(bool_cols),
            )
        elif (isinstance(axis, int) and axis == 1) or (isinstance(axis, str) and axis == "columns"):
            mask = None
            first = True
            for col in bool_cols:
                if first:
                    mask = self[col]
                    first = False
                else:
                    mask &= self[col]
            if first:
                mask = full(self.shape[0], True, dtype=bool)

            return Series(mask, index=self.index.values[:])
        elif axis is None:
            return all([akall(self[col].values) for col in bool_cols])
        else:
            raise ValueError("axis must have value 0, 1, 'index', 'columns', or None.")

    @typechecked
    def dropna(
        self,
        axis: Union[int, str] = 0,
        how: Optional[str] = None,
        thresh: Optional[int] = None,
        ignore_index: bool = False,
    ) -> DataFrame:
        """
        Remove missing values.

        Parameters
        ----------
        axis: {0 or 'index', 1 or 'columns'}, default = 0
            Determine if rows or columns which contain missing values are removed.

            0, or 'index': Drop rows which contain missing values.

            1, or 'columns': Drop columns which contain missing value.

            Only a single axis is allowed.
        how: {'any', 'all'}, default='any'
            Determine if row or column is removed from DataFrame, when we have at least one NA or all NA.

            'any': If any NA values are present, drop that row or column.

            'all': If all values are NA, drop that row or column.
        thresh: int, optional
            Require that many non - NA values.Cannot be combined with how.
        ignore_index: bool, default ``False``
            If ``True``, the resulting axis will be labeled 0, 1, …, n - 1.

        Returns
        -------
        arkouda.dataframe.DataFrame
            DataFrame with NA entries dropped from it.

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.connect()
        >>> import numpy as np
        >>> df = ak.DataFrame(
            {
                "A": [True, True, True, True],
                "B": [1, np.nan, 2, np.nan],
                "C": [1, 2, 3, np.nan],
                "D": [False, False, False, False],
                "E": [1, 2, 3, 4],
                "F": ["a", "b", "c", "d"],
                "G": [1, 2, 3, 4],
            }
           )

        >>> display(df)

        +----+------+-----+-----+-------+-----+-----+-----+
        |    | A    |   B |   C | D     |   E | F   |   G |
        +====+======+=====+=====+=======+=====+=====+=====+
        |  0 | True |   1 |   1 | False |   1 | a   |   1 |
        +----+------+-----+-----+-------+-----+-----+-----+
        |  1 | True | nan |   2 | False |   2 | b   |   2 |
        +----+------+-----+-----+-------+-----+-----+-----+
        |  2 | True |   2 |   3 | False |   3 | c   |   3 |
        +----+------+-----+-----+-------+-----+-----+-----+
        |  3 | True | nan | nan | False |   4 | d   |   4 |
        +----+------+-----+-----+-------+-----+-----+-----+

        >>> df.dropna()

        +----+------+-----+-----+-------+-----+-----+-----+
        |    | A    |   B |   C | D     |   E | F   |   G |
        +====+======+=====+=====+=======+=====+=====+=====+
        |  0 | True |   1 |   1 | False |   1 | a   |   1 |
        +----+------+-----+-----+-------+-----+-----+-----+
        |  1 | True |   2 |   3 | False |   3 | c   |   3 |
        +----+------+-----+-----+-------+-----+-----+-----+

        >>> df.dropna(axis=1)

        +----+------+-------+-----+-----+-----+
        |    | A    | D     |   E | F   |   G |
        +====+======+=======+=====+=====+=====+
        |  0 | True | False |   1 | a   |   1 |
        +----+------+-------+-----+-----+-----+
        |  1 | True | False |   2 | b   |   2 |
        +----+------+-------+-----+-----+-----+
        |  2 | True | False |   3 | c   |   3 |
        +----+------+-------+-----+-----+-----+
        |  3 | True | False |   4 | d   |   4 |
        +----+------+-------+-----+-----+-----+

        >>> df.dropna(axis=1, thresh=3)

        +----+------+-----+-------+-----+-----+-----+
        |    | A    |   C | D     |   E | F   |   G |
        +====+======+=====+=======+=====+=====+=====+
        |  0 | True |   1 | False |   1 | a   |   1 |
        +----+------+-----+-------+-----+-----+-----+
        |  1 | True |   2 | False |   2 | b   |   2 |
        +----+------+-----+-------+-----+-----+-----+
        |  2 | True |   3 | False |   3 | c   |   3 |
        +----+------+-----+-------+-----+-----+-----+
        |  3 | True | nan | False |   4 | d   |   4 |
        +----+------+-----+-------+-----+-----+-----+

        >>> df.dropna(axis=1, how="all")

        +----+------+-----+-----+-------+-----+-----+-----+
        |    | A    |   B |   C | D     |   E | F   |   G |
        +====+======+=====+=====+=======+=====+=====+=====+
        |  0 | True |   1 |   1 | False |   1 | a   |   1 |
        +----+------+-----+-----+-------+-----+-----+-----+
        |  1 | True | nan |   2 | False |   2 | b   |   2 |
        +----+------+-----+-----+-------+-----+-----+-----+
        |  2 | True |   2 |   3 | False |   3 | c   |   3 |
        +----+------+-----+-----+-------+-----+-----+-----+
        |  3 | True | nan | nan | False |   4 | d   |   4 |
        +----+------+-----+-----+-------+-----+-----+-----+

        """
        from arkouda import all as akall

        if (how is not None) and (thresh is not None):
            raise TypeError("You cannot set both the how and thresh arguments at the same time.")

        if how is None:
            how = "any"

        if (isinstance(axis, int) and axis == 0) or (isinstance(axis, str) and axis == "index"):
            agg_axis = 1

        elif (isinstance(axis, int) and axis == 1) or (isinstance(axis, str) and axis == "columns"):
            agg_axis = 0

        if thresh is not None:
            counts = self.count(axis=agg_axis)
            mask = counts >= thresh  # type: ignore
        elif how == "any":
            mask = self.notna().all(axis=agg_axis)
        elif how == "all":
            mask = self.notna().any(axis=agg_axis)
        else:
            raise ValueError(f"invalid how option: {how}")

        if (isinstance(mask, bool) and mask is True) or (
            isinstance(mask, Series) and akall(mask.values) is True
        ):
            result = self.copy(deep=None)
        else:
            if (isinstance(axis, int) and axis == 0) or (isinstance(axis, str) and axis == "index"):
                if self.empty is True:
                    result = DataFrame()
                else:
                    result = self[mask].copy(deep=True)
            elif (isinstance(axis, int) and axis == 1) or (isinstance(axis, str) and axis == "columns"):
                result = DataFrame()
                if isinstance(mask, Series):
                    for col, truth in zip(mask.index.values.to_list(), mask.values.to_list()):
                        if truth is True:
                            result[col] = self[col].values[:]

        if ignore_index is True and result.empty is False:
            result = result.reset_index()

        return result

    @typechecked
    def register(self, user_defined_name: str) -> DataFrame:
        """
        Register this DataFrame object and underlying components with the Arkouda server.

        Parameters
        ----------
        user_defined_name : str
            User defined name the DataFrame is to be registered under.
            This will be the root name for underlying components.

        Returns
        -------
        arkouda.dataframe.DataFrame
            The same DataFrame which is now registered with the arkouda server and has an updated name.
            This is an in-place modification, the original is returned to support a
            fluid programming style.
            Please note you cannot register two different DataFrames with the same name.

        Raises
        ------
        TypeError
            Raised if user_defined_name is not a str.
        RegistrationError
            If the server was unable to register the DataFrame with the user_defined_name.

        See also
        --------
        unregister
        attach
        unregister_dataframe_by_name
        is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.

        Any changes made to a DataFrame object after registering with the server may not be reflected
        in attached copies.

        Example
        -------

        >>> df = ak.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        >>> df.register("my_table_name")
        >>> df.attach("my_table_name")
        >>> df.is_registered()
        True
        >>> df.unregister()
        >>> df.is_registered()
        False


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
                "num_cols": len(self.columns.values),
                "column_names": self.columns.values,
                "columns": column_data,
                "col_objTypes": col_objTypes,
            },
        )
        self.registered_name = user_defined_name
        return self

    def unregister(self):
        """
        Unregister this DataFrame object in the arkouda server which was previously
        registered using register() and/or attached to using attach().

        Raises
        ------
        RegistrationError
            If the object is already unregistered or if there is a server error
            when attempting to unregister.

        See also
        --------
        register
        attach
        unregister_dataframe_by_name
        is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.

        Example
        -------

        >>> df = ak.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        >>> df.register("my_table_name")
        >>> df.attach("my_table_name")
        >>> df.is_registered()
        True
        >>> df.unregister()
        >>> df.is_registered()
        False

        """
        from arkouda.util import unregister

        if not self.registered_name:
            raise RegistrationError("This object is not registered")
        unregister(self.registered_name)
        self.registered_name = None  # Clear our internal DataFrame object name

    def is_registered(self) -> bool:
        """
        Return True if the object is contained in the registry.

        Returns
        -------
        bool
            Indicates if the object is contained in the registry.

        Raises
        ------
        RegistrationError
            Raised if there's a server-side error or a mismatch of registered components.

        See Also
        --------
        register
        attach
        unregister
        unregister_dataframe_by_name

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.

        Example
        -------

        >>> df = ak.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        >>> df.register("my_table_name")
        >>> df.attach("my_table_name")
        >>> df.is_registered()
        True
        >>> df.unregister()
        >>> df.is_registered()
        False


        """
        from arkouda.util import is_registered

        if self.registered_name is None:
            return False  # Dataframe cannot be registered as a component
        return is_registered(self.registered_name)

    @staticmethod
    def attach(user_defined_name: str) -> DataFrame:
        """
        Function to return a DataFrame object attached to the registered name in the
        arkouda server which was registered using register().

        Parameters
        ----------
        user_defined_name : str
            user defined name which DataFrame object was registered under.

        Returns
        -------
        arkouda.dataframe.DataFrame
               The DataFrame object created by re-attaching to the corresponding server components.

        Raises
        ------
        RegistrationError
            if user_defined_name is not registered

        See Also
        --------
        register, is_registered, unregister

        Example
        -------

        >>> df = ak.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        >>> df.register("my_table_name")
        >>> df.attach("my_table_name")
        >>> df.is_registered()
        True
        >>> df.unregister()
        >>> df.is_registered()
        False

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
    def unregister_dataframe_by_name(user_defined_name: str) -> str:
        """
        Function to unregister DataFrame object by name which was registered
        with the arkouda server via register().

        Parameters
        ----------
        user_defined_name : str
            Name under which the DataFrame object was registered.

        Raises
        -------
        TypeError
            If user_defined_name is not a string.
        RegistrationError
            If there is an issue attempting to unregister any underlying components.

        See Also
        --------
        register
        unregister
        attach
        is_registered

        Example
        -------

        >>> df = ak.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        >>> df.register("my_table_name")
        >>> df.attach("my_table_name")
        >>> df.is_registered()
        True
        >>> df.unregister_dataframe_by_name("my_table_name")
        >>> df.is_registered()
        False

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
        tuple
            (columnName, columnType)
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
        """
        Creates a DataFrame object from an arkouda server response message.

        Parameters
        ----------
        rep_msg : string
            Server response message used to create a DataFrame.

        Returns
        -------
        arkouda.dataframe.DataFrame

        """
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


class _LocIndexer:
    def __init__(self, df):
        self.df = df

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            return self._get_row_col(key[0], key[1])
        if isinstance(key, list):
            key = array(key)
        if isinstance(key, Series):
            key = key.values
        if is_supported_scalar(key) and self.df.index.dtype == dtype(type(key)):
            return self.df._get_rows(indexof1d(array([key]), self.df.index.values))

        if isinstance(key, pdarray) and key.dtype == self.df.index.dtype:
            return self.df._get_rows(indexof1d(key, self.df.index.values))

        if isinstance(key, slice):
            if key.start is not None and akfind(array([key.start]), self.df.index.values)[0] == -1:
                raise KeyError(f"Index {key.start} not found in DataFrame index")
            if key.stop is not None and akfind(array([key.stop]), self.df.index.values)[0] == -1:
                raise KeyError(f"Index {key.stop} not found in DataFrame index")

            start_idx = (
                akfind(array([key.start]), self.df.index.values)[0] if key.start is not None else 0
            )
            stop_idx = (
                akfind(array([key.stop]), self.df.index.values)[0] + 1
                if key.stop is not None
                else self.df.index.size
            )

            indices = arange(start_idx, stop_idx)
            return self.df._get_rows(indices)

        if isinstance(key, pdarray) and key.dtype == akbool:
            return self.df._get_rows(key)

        return None

    def _get_row_col(self, row_key, col_key):
        return self[row_key][col_key]

    def __setitem__(self, key, val):
        if isinstance(key, tuple) and len(key) == 2:
            self._set_row_col(key[0], key[1], val)
            return
        else:
            raise ValueError(
                "Invalid key type. '.loc' indexing only supports keys with row and column selectors."
            )

    def _set_row_col(self, row_key, col_key, val):
        if isinstance(row_key, list):
            row_key = array(row_key)
        if isinstance(row_key, Series):
            row_key = row_key.values
        if is_supported_scalar(col_key) and col_key not in self.df.columns.values:
            self.df._add_column(col_key, dtype(type(val)))

        if is_supported_scalar(val):
            return self._set_row_col_scalar_val(row_key, col_key, val)
        else:
            assert isinstance(val, (pdarray, Series, Strings, SegArray)), "Invalid value type"
            return self._set_row_col_vector_val(row_key, col_key, val)

    def _set_row_col_scalar_val(self, row_key, col_key, val):
        if is_supported_scalar(row_key):
            if not self.df.index.dtype == dtype(type(row_key)):
                raise TypeError("Row key must be of the same type as the DataFrame index")
            if akfind(array([row_key]), self.df.index.values)[0] == -1:
                self.df._add_new_rows(row_key)
            # updating a single row
            row_idx = indexof1d(array([row_key]), self.df.index.values)
            if row_idx.size == 0:
                raise ValueError(f"Index {row_key} not found in DataFrame index")

            self.df.data[col_key][row_idx] = val

        if isinstance(row_key, pdarray) and row_key.dtype == self.df.index.dtype:
            if akany(in1d(row_key, self.df.index.values, invert=True)):
                self.df._add_new_rows(row_key)
            # updating multiple rows
            row_idx = indexof1d(row_key, self.df.index.values)
            self.df.data[col_key][row_idx] = val

        if isinstance(row_key, pdarray) and row_key.dtype == akbool:
            self.df.data[col_key][row_key] = val
        if isinstance(row_key, slice):
            if (
                row_key.start is not None
                and akfind(array([row_key.start]), self.df.index.values)[0] == -1
            ):
                raise KeyError(f"Index {row_key.start} not found in DataFrame index")
            if row_key.stop is not None and akfind(array([row_key.stop]), self.df.index.values)[0] == -1:
                raise KeyError(f"Index {row_key.stop} not found in DataFrame index")

            start_idx = (
                akfind(array([row_key.start]), self.df.index.values)[0]
                if row_key.start is not None
                else 0
            )
            stop_idx = (
                akfind(array([row_key.stop]), self.df.index.values)[0] + 1
                if row_key.stop is not None
                else self.df.index.size
            )
            indices = arange(start_idx, stop_idx)
            self.df.data[col_key][indices] = val
        return None

    def _set_row_col_vector_val(self, row_key, col_key, val):
        if isinstance(val, Series):
            aligned_indices = indexof1d(val.index.values, self.df.index.values)
            self.df.data[col_key][aligned_indices] = val.values
            return
        if isinstance(row_key, pdarray) and row_key.dtype == self.df.index.dtype:
            if akany(in1d(row_key, self.df.index.values, invert=True)):
                self.df._add_new_rows(row_key)
            # updating multiple rows
            row_idx = indexof1d(row_key, self.df.index.values)
            self.df.data[col_key][row_idx] = val
        if isinstance(row_key, slice):
            if (
                row_key.start is not None
                and akfind(array([row_key.start]), self.df.index.values)[0] == -1
            ):
                raise ValueError(f"Index {row_key.start} not found in DataFrame index")
            if row_key.stop is not None and akfind(array([row_key.stop]), self.df.index.values)[0] == -1:
                raise ValueError(f"Index {row_key.stop} not found in DataFrame index")

            start_idx = (
                akfind(array([row_key.start]), self.df.index.values)[0]
                if row_key.start is not None
                else 0
            )
            # should the below have + 1 like the other stop_idxs?
            stop_idx = (
                akfind(array([row_key.stop]), self.df.index.values)[0]
                if row_key.stop is not None
                else self.df.index.size
            )

            indices = arange(start_idx, stop_idx)
            self.df.data[col_key][indices] = val
        return None


class _ILocIndexer:
    def __init__(self, df):
        self.df = df

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            return self._get_row_col(key[0], key[1])
        if isinstance(key, list):
            key = array(key)
        if isinstance(key, Series):
            key = key.values

        if is_supported_scalar(key):
            if not isinstance(key, int):
                raise TypeError("iloc key must be an integer")
            if key >= len(self.df) or key < -len(self.df):
                raise IndexError("Index out of range")
            return self.df._get_rows(array([key]))

        if isinstance(key, pdarray):
            if key.dtype == akint64:
                if akany(key < -len(self.df)) or akany(key >= len(self.df)):
                    raise IndexError("Index out of range")
                return self.df._get_rows(key)
            if key.dtype == akbool:
                if key.size != self.df.index.size:
                    raise IndexError("Boolean array must be the same size as the DataFrame index")
                return self.df._get_rows(key)
            raise TypeError("Invalid dtype for iloc key, must be int or bool: {}".format(key.dtype))

        if isinstance(key, slice):
            if key.start is not None and not isinstance(key.start, int):
                raise TypeError("Start of slice must be an integer")
            if key.stop is not None and not isinstance(key.stop, int):
                raise TypeError("Stop of slice must be an integer")
            if key.step is not None and not isinstance(key.step, int):
                raise TypeError("Step of slice must be an integer")
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self.df.index.size
            step = key.step if key.step is not None else 1
            if start < 0 or start >= len(self.df) or stop < 0 or stop > len(self.df) or step <= 0:
                raise IndexError("Slice index out of range")
            return self.df._get_rows(arange(start, stop, step))

        raise TypeError("Invalid iloc key: {}".format(key))

    def _get_row_col(self, row_key, col_key):
        row_indexed = self[row_key]

        if isinstance(col_key, list):
            col_key = array(col_key)
        if isinstance(col_key, Series):
            col_key = col_key.values

        if isinstance(row_indexed, DataFrame):
            if isinstance(col_key, int):
                column_name = row_indexed.columns[col_key]
                column = row_indexed[column_name]
                if len(column) == 1:
                    return column.values[0]
                else:
                    return column

            if isinstance(col_key, pdarray):
                column_array = array(row_indexed.columns)
                if col_key.dtype == akbool or col_key.dtype == akint64:
                    return row_indexed[column_array[col_key]]
                raise ValueError(
                    "Invalid dtype for iloc key, must be int or bool: {}".format(col_key.dtype)
                )

    def __setitem__(self, key, val):
        if isinstance(key, tuple) and len(key) == 2:
            self._set_row_col(key[0], key[1], val)
            return
        else:
            raise ValueError(
                "Invalid key type. '.iloc' indexing only supports keys with row and column selectors."
            )

    def _set_row_col(self, row_key, col_key, val):
        if isinstance(row_key, list):
            row_key = array(row_key)
        if isinstance(row_key, Series):
            row_key = row_key.values

        # Only supports setting a single column at a time
        if not isinstance(col_key, int):
            raise ValueError("Column key must be an integer")
        if col_key >= len(self.df.columns) or col_key < -len(self.df.columns):
            raise IndexError("Index out of range")

        row_indices = None

        if is_supported_scalar(row_key):
            if not isinstance(row_key, int):
                raise ValueError("Row key must be an integer")
            if row_key >= len(self.df) or row_key < -len(self.df):
                raise IndexError("Index out of range")
            row_indices = array([row_key])
        elif isinstance(row_key, pdarray):
            if row_key.dtype == akint64:
                if akany(row_key < -len(self.df)) or akany(row_key >= len(self.df)):
                    raise IndexError("Index out of range")
                row_indices = row_key
            elif row_key.dtype == akbool:
                if row_key.size != self.df.index.size:
                    raise IndexError("Boolean array must be the same size as the DataFrame index")
                row_indices = row_key
            else:
                raise ValueError(
                    "Invalid dtype for iloc key, must be int or bool: {}".format(row_key.dtype)
                )
        elif isinstance(row_key, slice):
            if row_key.start is not None and not isinstance(row_key.start, int):
                raise ValueError("Start of slice must be an integer")
            if row_key.stop is not None and not isinstance(row_key.stop, int):
                raise ValueError("Stop of slice must be an integer")
            if row_key.step is not None and not isinstance(row_key.step, int):
                raise ValueError("Step of slice must be an integer")
            start = row_key.start if row_key.start is not None else 0
            stop = row_key.stop if row_key.stop is not None else self.df.index.size
            step = row_key.step if row_key.step is not None else 1
            row_indices = arange(start, stop, step)
        else:
            raise TypeError("invalid row key type: {}".format(type(row_key)))

        if is_supported_scalar(val):
            self.df.data[self.df.columns[col_key]][row_indices] = val
        elif isinstance(val, pdarray):
            if val.size != len(row_indices):
                raise ValueError("Value array must be the same size as the row indices")
            self.df.data[self.df.columns[col_key]][row_indices] = val
        else:
            raise ValueError("Invalid value type: {}".format(type(val)))


class AtIndexer:
    def __init__(self, df) -> None:
        self.df = df

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(".at requires a row key and a column key")
        (row, col) = key
        if not is_supported_scalar(row):
            raise ValueError(".at only supports scalar row keys")
        if not is_supported_scalar(col):
            raise ValueError(".at only supports scalar column keys")
        return self.df.loc[row, col]

    def __setitem__(self, key, val):
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(".at requires a row key and a column key")
        (row, col) = key
        if not is_supported_scalar(row):
            raise ValueError(".at only supports scalar row keys")
        if not is_supported_scalar(col):
            raise ValueError(".at only supports scalar column keys")
        self.df.loc[row, col] = val


class IAtIndexer:
    def __init__(self, df) -> None:
        self.df = df

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(".iat requires a row key and a column key")
        (row, col) = key
        if not isinstance(row, int):
            raise ValueError(".iat requires integer row keys")
        if not isinstance(col, int):
            raise ValueError(".iat requires integer column keys")
        return self.df.iloc[row, col]

    def __setitem__(self, key, val):
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(".iat requires a row key and a column key")
        (row, col) = key
        if not is_supported_scalar(row):
            raise ValueError(".iat requires integer row keys")
        if not is_supported_scalar(col):
            raise ValueError(".iat requires integer column keys")
        self.df.iloc[row, col] = val


def intx(a, b):
    """
    Find all the rows that are in both dataframes.
    Columns should be in identical order.

    Note: does not work for columns of floating point values, but does work for
    Strings, pdarrays of int64 type, and Categorical *should* work.

    Examples
    --------

    >>> import arkouda as ak
    >>> ak.connect()
    >>> a = ak.DataFrame({'a':ak.arange(5),'b': 2* ak.arange(5)})
    >>> display(a)

    +----+-----+-----+
    |    |   a |   b |
    +====+=====+=====+
    |  0 |   0 |   0 |
    +----+-----+-----+
    |  1 |   1 |   2 |
    +----+-----+-----+
    |  2 |   2 |   4 |
    +----+-----+-----+
    |  3 |   3 |   6 |
    +----+-----+-----+
    |  4 |   4 |   8 |
    +----+-----+-----+

    >>> b = ak.DataFrame({'a':ak.arange(5),'b':ak.array([0,3,4,7,8])})
    >>> display(b)

    +----+-----+-----+
    |    |   a |   b |
    +====+=====+=====+
    |  0 |   0 |   0 |
    +----+-----+-----+
    |  1 |   1 |   3 |
    +----+-----+-----+
    |  2 |   2 |   4 |
    +----+-----+-----+
    |  3 |   3 |   7 |
    +----+-----+-----+
    |  4 |   4 |   8 |
    +----+-----+-----+

    >>> intx(a,b)
    >>> intersect_df = a[intx(a,b)]
    >>> display(intersect_df)

    +----+-----+-----+
    |    |   a |   b |
    +====+=====+=====+
    |  0 |   0 |   0 |
    +----+-----+-----+
    |  1 |   2 |   4 |
    +----+-----+-----+
    |  2 |   4 |   8 |
    +----+-----+-----+

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
    a : Strings or pdarray
        An array of strings.

    b : Strings or pdarray
        An array of strings.

    positions : bool, default=True
        Return tuple of boolean pdarrays that indicate positions in `a` and `b`
        of the intersection values.

    unique : bool, default=False
        If the number of distinct values in `a` (and `b`) is equal to the size of
        `a` (and `b`), there is a more efficient method to compute the intersection.

    Returns
    -------
    (arkouda.pdarrayclass.pdarray, arkouda.pdarrayclass.pdarray) or arkouda.pdarrayclass.pdarray
        The indices of `a` and `b` where any element occurs at least once in both
        arrays.

    Examples
    --------

    >>> import arkouda as ak
    >>> ak.connect()
    >>> a = ak.arange(10)
    >>> print(a)
    [0 1 2 3 4 5 6 7 8 9]

    >>> b = 2 * ak.arange(10)
    >>> print(b)
    [0 2 4 6 8 10 12 14 16 18]

    >>> intersect(a,b, positions=True)
    (array([True False True False True False True False True False]),
    array([True True True True True False False False False False]))

    >>> intersect(a,b, positions=False)
    array([0 2 4 6 8])

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
    perm : pdarray
        The permutation array.

    Returns
    -------
    arkouda.pdarrayclass.pdarray
        The inverse of the permutation array.

    Examples
    --------

    >>> import arkouda as ak
    >>> ak.connect()
    >>> from arkouda.index import Index
    >>> i = Index(ak.array([1,2,0,5,4]))
    >>> perm = i.argsort()
    >>> print(perm)
    [2 0 1 4 3]
    >>> invert_permutation(perm)
    array([1 2 0 4 3])

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
    sort: bool = True,
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
    sort: bool = True
        If True, DataFrame is returned sorted by "on".
        Otherwise, the DataFrame is not sorted.
    Returns
    -------
    arkouda.dataframe.DataFrame
        Inner-Joined Arkouda DataFrame
    """
    left_cols, right_cols = left.columns.values.copy(), right.columns.values.copy()
    if isinstance(on, str):
        left_inds, right_inds = inner_join(left[on].values, right[on].values)
        new_dict = {on: left[on].iloc[left_inds]}
        left_cols.remove(on)
        right_cols.remove(on)
    else:
        left_inds, right_inds = inner_join(
            [left[col].values for col in on], [right[col].values for col in on]
        )
        new_dict = {col: left[col].iloc[left_inds] for col in on}
        for col in on:
            left_cols.remove(col)
            right_cols.remove(col)

    for col in left_cols:
        new_col = col + left_suffix if col in col_intersect else col
        new_dict[new_col] = left[col].iloc[left_inds]
    for col in right_cols:
        new_col = col + right_suffix if col in col_intersect else col
        new_dict[new_col] = right[col].iloc[right_inds]

    ret_df = DataFrame(new_dict)
    if sort is True:
        ret_df = ret_df.sort_values(on).reset_index()
    return ret_df


def _right_join_merge(
    left: DataFrame,
    right: DataFrame,
    on: Union[str, List[str]],
    col_intersect: Union[str, List[str]],
    left_suffix: str = "_x",
    right_suffix: str = "_y",
    convert_ints: bool = True,
    sort: bool = True,
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
    convert_ints: bool = True
        If True, convert columns with missing int values (due to the join) to float64.
        This is to match pandas.
        If False, do not convert the column dtypes.
    sort: bool = True
        If True, DataFrame is returned sorted by "on".
        Otherwise, the DataFrame is not sorted.
    Returns
    -------
    arkouda.dataframe.DataFrame
        Right-Joined Arkouda DataFrame
    """
    in_left = _inner_join_merge(left, right, on, col_intersect, left_suffix, right_suffix, sort=False)
    in_left_cols, left_cols = in_left.columns.values.copy(), left.columns.values.copy()
    if isinstance(on, str):
        left_at_on = left[on].values
        right_at_on = right[on].values
        left_cols.remove(on)
        in_left_cols.remove(on)
    else:
        left_at_on = [left[col].values for col in on]
        right_at_on = [right[col].values for col in on]
        for col in on:
            left_cols.remove(col)
            in_left_cols.remove(col)

    not_in_left = right[in1d(right_at_on, left_at_on, invert=True)]
    for col in not_in_left.columns:
        if col in left_cols:
            not_in_left[col + right_suffix] = not_in_left[col]
            not_in_left = not_in_left.drop(col, axis=1)

    nan_cols = list(set(in_left) - set(not_in_left))
    for col in nan_cols:
        if convert_ints is True and in_left[col].dtype == int:
            in_left[col] = akcast(in_left[col].values, akfloat64)

        # Create a nan array for all values not in the left df
        not_in_left[col] = __nulls_like(in_left[col].values, len(not_in_left))
    ret_df = DataFrame.append(in_left, not_in_left)
    if sort is True:
        ret_df = ret_df.sort_values(on).reset_index()
    return ret_df


def _outer_join_merge(
    left: DataFrame,
    right: DataFrame,
    on: Union[str, List[str]],
    col_intersect: Union[str, List[str]],
    left_suffix: str = "_x",
    right_suffix: str = "_y",
    convert_ints: bool = True,
    sort: bool = True,
) -> DataFrame:
    """
    Utilizes the ak.join.inner_join_merge function to return an
    ak DataFrame object containing all the rows in each DataFrame (based on the "on" param),
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
    convert_ints: bool = True
        If True, convert columns with missing int values (due to the join) to float64.
        This is to match pandas.
        If False, do not convert the column dtypes.
    sort: bool = True
        If True, DataFrame is returned sorted by "on".
        Otherwise, the DataFrame is not sorted.
    Returns
    -------
    arkouda.dataframe.DataFrame
        Outer-Joined Arkouda DataFrame
    """
    inner = _inner_join_merge(left, right, on, col_intersect, left_suffix, right_suffix, sort=False)
    left_cols, right_cols = (
        left.columns.values.copy(),
        right.columns.values.copy(),
    )

    if isinstance(on, str):
        left_at_on = left[on].values
        right_at_on = right[on].values
        left_cols.remove(on)
        right_cols.remove(on)

    else:
        left_at_on = [left[col].values for col in on]
        right_at_on = [right[col].values for col in on]
        for col in on:
            left_cols.remove(col)
            right_cols.remove(col)

    not_in_left = right[in1d(right_at_on, left_at_on, invert=True)]
    for col in not_in_left.columns:
        if col in left_cols:
            not_in_left[col + right_suffix] = not_in_left[col]
            not_in_left = not_in_left.drop(col, axis=1)

    not_in_right = left[in1d(left_at_on, right_at_on, invert=True)]
    for col in not_in_right.columns:
        if col in right_cols:
            not_in_right[col + left_suffix] = not_in_right[col]
            not_in_right = not_in_right.drop(col, axis=1)

    left_nan_cols = list(set(inner) - set(not_in_left))
    right_nan_cols = list(set(inner) - set(not_in_right))

    for col in set(left_nan_cols).union(set(right_nan_cols)):
        if convert_ints is True and inner[col].dtype == int:
            inner[col] = akcast(inner[col].values, akfloat64)
        if col in left_nan_cols:
            if convert_ints is True and not_in_right[col].dtype == int:
                not_in_right[col] = akcast(not_in_right[col].values, akfloat64)
            elif col in not_in_left.columns.values:
                not_in_right[col] = akcast(not_in_right[col].values, not_in_left[col].dtype)
        if col in right_nan_cols:
            if convert_ints is True and not_in_left[col].dtype == int:
                not_in_left[col] = akcast(not_in_left[col].values, akfloat64)
            elif col in not_in_right.columns.values:
                not_in_left[col] = akcast(not_in_left[col].values, not_in_right[col].dtype)

    for col in left_nan_cols:
        # Create a nan array for all values not in the left df
        not_in_left[col] = __nulls_like(inner[col].values, len(not_in_left))

    for col in right_nan_cols:
        # Create a nan array for all values not in the left df
        not_in_right[col] = __nulls_like(inner[col].values, len(not_in_right))

    ret_df = DataFrame.append(DataFrame.append(inner, not_in_left), not_in_right)
    if sort is True:
        ret_df = ret_df.sort_values(on).reset_index()

    return ret_df


def __nulls_like(
    arry: Union[pdarray, Strings, Categorical],
    size: Optional[
        Union[
            int,
            np.signedinteger[_8Bit],
            np.signedinteger[_16Bit],
            np.signedinteger[_32Bit],
            np.signedinteger[_64Bit],
            np.unsignedinteger[_8Bit],
            np.unsignedinteger[_16Bit],
            np.unsignedinteger[_32Bit],
            np.unsignedinteger[_64Bit],
        ]
    ] = None,
):
    if size is None:
        size = arry.size

    if isinstance(arry, (Strings, Categorical)):
        return full(size, "nan")
    else:
        return full(size, np.nan, arry.dtype)


@typechecked
def merge(
    left: DataFrame,
    right: DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    how: str = "inner",
    left_suffix: str = "_x",
    right_suffix: str = "_y",
    convert_ints: bool = True,
    sort: bool = True,
) -> DataFrame:
    r"""
    Merge Arkouda DataFrames with a database-style join.
    The resulting dataframe contains rows from both DataFrames as specified by
    the merge condition (based on the "how" and "on" parameters).

    Based on pandas merge functionality.
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html

    Parameters
    ----------
    left: DataFrame
        The Left DataFrame to be joined.
    right: DataFrame
        The Right DataFrame to be joined.
    on: Optional[Union[str, List[str]]] = None
        The name or list of names of the DataFrame column(s) to join on.
        If on is None, this defaults to the intersection of the columns in both DataFrames.
    how: str, default = "inner"
        The merge condition.
        Must be one of "inner", "left", "right", or "outer".
    left_suffix: str, default = "_x"
        A string indicating the suffix to add to columns from the left dataframe for overlapping
        column names in both left and right. Defaults to "_x". Only used when how is "inner".
    right_suffix: str, default = "_y"
        A string indicating the suffix to add to columns from the right dataframe for overlapping
        column names in both left and right. Defaults to "_y". Only used when how is "inner".
    convert_ints: bool = True
        If True, convert columns with missing int values (due to the join) to float64.
        This is to match pandas.
        If False, do not convert the column dtypes.
        This has no effect when how = "inner".
    sort: bool = True
        If True, DataFrame is returned sorted by "on".
        Otherwise, the DataFrame is not sorted.
    Returns
    -------
    arkouda.dataframe.DataFrame
        Joined Arkouda DataFrame.

    Note
    ----
    Multiple column joins are only supported for integer columns.

    Examples
    --------

    >>> import arkouda as ak
    >>> ak.connect()
    >>> from arkouda import merge
    >>> left_df = ak.DataFrame({'col1': ak.arange(5), 'col2': -1 * ak.arange(5)})
    >>> display(left_df)

    +----+--------+--------+
    |    |   col1 |   col2 |
    +====+========+========+
    |  0 |      0 |      0 |
    +----+--------+--------+
    |  1 |      1 |     -1 |
    +----+--------+--------+
    |  2 |      2 |     -2 |
    +----+--------+--------+
    |  3 |      3 |     -3 |
    +----+--------+--------+
    |  4 |      4 |     -4 |
    +----+--------+--------+

    >>> right_df = ak.DataFrame({'col1': 2 * ak.arange(5), 'col2': 2 * ak.arange(5)})
    >>> display(right_df)

    +----+--------+--------+
    |    |   col1 |   col2 |
    +====+========+========+
    |  0 |      0 |      0 |
    +----+--------+--------+
    |  1 |      2 |      2 |
    +----+--------+--------+
    |  2 |      4 |      4 |
    +----+--------+--------+
    |  3 |      6 |      6 |
    +----+--------+--------+
    |  4 |      8 |      8 |
    +----+--------+--------+

    >>> merge(left_df, right_df, on = "col1")

    +----+--------+----------+----------+
    |    |   col1 |   col2_x |   col2_y |
    +====+========+==========+==========+
    |  0 |      0 |        0 |        0 |
    +----+--------+----------+----------+
    |  1 |      2 |       -2 |        2 |
    +----+--------+----------+----------+
    |  2 |      4 |       -4 |        4 |
    +----+--------+----------+----------+

    >>> merge(left_df, right_df, on = "col1", how = "left")

    +----+--------+----------+----------+
    |    |   col1 |   col2_y |   col2_x |
    +====+========+==========+==========+
    |  0 |      0 |        0 |        0 |
    +----+--------+----------+----------+
    |  1 |      1 |      nan |       -1 |
    +----+--------+----------+----------+
    |  2 |      2 |        2 |       -2 |
    +----+--------+----------+----------+
    |  3 |      3 |      nan |       -3 |
    +----+--------+----------+----------+
    |  4 |      4 |        4 |       -4 |
    +----+--------+----------+----------+

    >>> merge(left_df, right_df, on = "col1", how = "right")

    +----+--------+----------+----------+
    |    |   col1 |   col2_x |   col2_y |
    +====+========+==========+==========+
    |  0 |      0 |        0 |        0 |
    +----+--------+----------+----------+
    |  1 |      2 |       -2 |        2 |
    +----+--------+----------+----------+
    |  2 |      4 |       -4 |        4 |
    +----+--------+----------+----------+
    |  3 |      6 |      nan |        6 |
    +----+--------+----------+----------+
    |  4 |      8 |      nan |        8 |
    +----+--------+----------+----------+

    >>> merge(left_df, right_df, on = "col1", how = "outer")

    +----+--------+----------+----------+
    |    |   col1 |   col2_y |   col2_x |
    +====+========+==========+==========+
    |  0 |      0 |        0 |        0 |
    +----+--------+----------+----------+
    |  1 |      1 |      nan |       -1 |
    +----+--------+----------+----------+
    |  2 |      2 |        2 |       -2 |
    +----+--------+----------+----------+
    |  3 |      3 |      nan |       -3 |
    +----+--------+----------+----------+
    |  4 |      4 |        4 |       -4 |
    +----+--------+----------+----------+
    |  5 |      6 |        6 |      nan |
    +----+--------+----------+----------+
    |  6 |      8 |        8 |      nan |
    +----+--------+----------+----------+

    """
    col_intersect = list(set(left.columns) & set(right.columns))
    on = on if on is not None else col_intersect

    if not isinstance(on, str):
        if not all(
            isinstance(left[col].values, (pdarray, Strings))
            and isinstance(right[col].values, (pdarray, Strings))
            for col in on
        ):
            raise ValueError("All columns of a multi-column merge must be pdarrays")

    if how == "inner":
        return _inner_join_merge(left, right, on, col_intersect, left_suffix, right_suffix, sort=sort)
    elif how == "right":
        return _right_join_merge(
            left,
            right,
            on,
            col_intersect,
            left_suffix,
            right_suffix,
            convert_ints=convert_ints,
            sort=sort,
        )
    elif how == "left":
        return _right_join_merge(
            right,
            left,
            on,
            col_intersect,
            right_suffix,
            left_suffix,
            convert_ints=convert_ints,
            sort=sort,
        )
    elif how == "outer":
        warn(
            "Outer joins should not be performed on large data sets as they may require "
            "prohibitive amounts of memory.",
            UserWarning,
        )
        return _outer_join_merge(
            right,
            left,
            on,
            col_intersect,
            right_suffix,
            left_suffix,
            convert_ints=convert_ints,
            sort=sort,
        )
    else:
        raise ValueError(
            f"Unexpected value of {how} for how. Must choose: 'inner', 'left', 'right' or 'outer'"
        )
